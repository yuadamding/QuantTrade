from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TradingConstraintConfig:
    max_switches_per_day: int | None = None
    max_switches_per_episode: int | None = None
    max_order_legs_per_day: float | None = None
    max_order_legs_per_episode: float | None = None
    min_hold_bars: int = 1
    cooldown_bars: int = 0
    q_switch_margin_bps: float = 0.0
    extra_switch_penalty_bps: float = 0.0
    one_way_cost_bps: float = 1.0
    count_etf_to_etf_as_two_legs: bool = True
    cash_index: int = 0


CONSTRAINT_FEATURE_DIM = 6
CONSTRAINED_POLICY_MODEL_VERSION = 2
# Position-aware policy: scores each candidate with the (held position -> candidate) transition.
# A distinct model contract so a transition-aware checkpoint cannot be confused with a v2 one.
POSITION_AWARE_POLICY_MODEL_VERSION = 3
TRANSITION_FEATURE_SCHEMA_VERSION = 1
CONSTRAINT_FEATURE_NAMES = [
    "bars_held_over_min_hold",
    "cooldown_remaining_over_cooldown",
    "switches_today_over_cap",
    "switches_episode_over_cap",
    "order_legs_today_over_cap",
    "order_legs_episode_over_cap",
]


def _cap_denominator(value: int | float | None, fallback: int | float) -> float:
    raw = fallback if value is None else value
    return max(float(raw), 1.0)


def trade_legs(
    previous_action: torch.Tensor,
    action: torch.Tensor,
    *,
    cash_index: int = 0,
    count_etf_to_etf_as_two_legs: bool = True,
) -> torch.Tensor:
    changed = action != previous_action
    if not count_etf_to_etf_as_two_legs:
        return changed.float()
    prev_risky = previous_action != int(cash_index)
    next_risky = action != int(cash_index)
    legs = torch.zeros_like(action, dtype=torch.float32)
    legs = torch.where(changed & prev_risky, legs + 1.0, legs)
    legs = torch.where(changed & next_risky, legs + 1.0, legs)
    return legs


TRANSITION_FEATURE_NAMES = [
    "is_hold",
    "is_switch",
    "prev_is_cash",
    "cand_is_cash",
    "legs",
    "est_cost_bps_over_100",
    "leverage_delta",
    "same_group",
]
TRANSITION_FEATURE_DIM = len(TRANSITION_FEATURE_NAMES)

# PR-D dynamic position state (opt-in, behind use_dynamic_transition_features; D1 lands the builder/constants
# only -- nothing consumes them until D3). These columns carry the HELD position's realized-P&L EXCURSION,
# the signal the static [A,A,F] table and the constraint_features counters do NOT contain, so a
# position-aware Q can weigh hold-vs-exit by how the open position has actually performed. This env is
# RETURN-based (no prices / target weights), so the design's price-relative and executed-weight columns are
# intentionally absent; the holding/cooldown/switch/leg counters are already in constraint_features and are
# deliberately NOT duplicated here.
DYNAMIC_TRANSITION_FEATURE_SCHEMA_VERSION = 2
# A distinct model contract so a dynamic-transition-aware checkpoint (wider input) cannot be confused with a
# v3 static-transition one at load.
DYNAMIC_POSITION_AWARE_POLICY_MODEL_VERSION = 4
DYNAMIC_TRANSITION_FEATURE_NAMES = [
    "unrealized_pnl",  # compounded return since entry
    "max_adverse_excursion",  # most-negative cumulative return since entry (<= 0)
    "max_favorable_excursion",  # most-positive cumulative return since entry (>= 0)
    "drawdown_from_peak",  # max_favorable_excursion - unrealized_pnl (>= 0)
    "runup_from_trough",  # unrealized_pnl - max_adverse_excursion (>= 0)
]
DYNAMIC_TRANSITION_FEATURE_DIM = len(DYNAMIC_TRANSITION_FEATURE_NAMES)


def build_dynamic_transition_features(
    *,
    unrealized_pnl: torch.Tensor,
    mae: torch.Tensor,
    mfe: torch.Tensor,
    clamp: float = 1.0,
) -> torch.Tensor:
    """Per-env ``[B, DYNAMIC_TRANSITION_FEATURE_DIM]`` dynamic position-state features (PR-D D1).

    Built ONLY from the env's RETURN-based held-position bookkeeping (no prices/weights): the compounded
    return since entry (``unrealized_pnl``), its max adverse / favorable excursion, and the derived
    drawdown-from-peak and run-up-from-trough. Returns are clamped to +/-``clamp`` (derived spreads to
    ``2*clamp``) so a runaway compounded value can't dominate the encoder -- mirroring
    ``make_constraint_features`` normalize+clamp discipline. The block is per-env (independent of the
    candidate action); the forward broadcasts it across candidates. Pure/deterministic; no model state."""
    band = float(clamp)
    upnl = unrealized_pnl.float().clamp(-band, band)
    adverse = mae.float().clamp(-band, band)
    favorable = mfe.float().clamp(-band, band)
    drawdown = (favorable - upnl).clamp(0.0, 2.0 * band)
    runup = (upnl - adverse).clamp(0.0, 2.0 * band)
    return torch.stack([upnl, adverse, favorable, drawdown, runup], dim=1)


def advance_position_excursion(
    unrealized_pnl: torch.Tensor,
    mae: torch.Tensor,
    mfe: torch.Tensor,
    raw_return: torch.Tensor,
    *,
    held: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance the held position's (unrealized P&L, MAE, MFE) by one step's raw (gross) return.

    On a HOLD (``held``) compound this step's return into the position-since-entry and extend the max
    adverse / favorable excursion; on a SWITCH the new position starts fresh this step. Tensor-shaped, so it
    serves both the vectorized env step and a scalar evaluation rollout (1-element tensors). This is the
    SINGLE source of truth for the excursion recurrence, so the env and the evaluator cannot drift; the env
    additionally tracks ``entry_index`` (not needed to build the dynamic features)."""
    cum = torch.where(held, (1.0 + unrealized_pnl) * (1.0 + raw_return) - 1.0, raw_return)
    zeros = torch.zeros_like(cum)
    mae = torch.where(held, torch.minimum(mae, cum), torch.minimum(zeros, cum))
    mfe = torch.where(held, torch.maximum(mfe, cum), torch.maximum(zeros, cum))
    return cum, mae, mfe


def build_transition_feature_table(
    *,
    action_count: int,
    cash_index: int,
    one_way_cost_bps: float,
    extra_switch_penalty_bps: float = 0.0,
    count_etf_to_etf_as_two_legs: bool = True,
    action_leverage: torch.Tensor,
    action_group_ids: torch.Tensor,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Static ``[A, A, TRANSITION_FEATURE_DIM]`` table of (previous_action, candidate_action) features.

    Dim 0 indexes the HELD/previous action, dim 1 the CANDIDATE action. Every feature is a deterministic
    function of ``(prev, cand)`` and static action metadata -- the legs/cost columns mirror the env reward
    exactly (``legs * one_way_cost_bps + switch * extra_switch_penalty_bps``) -- so there is no market,
    reward, or future-label leakage and the table can be gathered by ``previous_action`` id inside the
    Q-network: ``table[previous_actions]`` gives the ``[B, A, F]`` per-candidate transition tensor."""
    device = action_leverage.device if device is None else device
    leverage = action_leverage.to(device=device, dtype=torch.float32)
    groups = action_group_ids.to(device=device)
    arange = torch.arange(action_count, device=device)
    prev = arange[:, None].expand(action_count, action_count)
    cand = arange[None, :].expand(action_count, action_count)
    is_switch = cand != prev
    legs = trade_legs(prev, cand, cash_index=cash_index, count_etf_to_etf_as_two_legs=count_etf_to_etf_as_two_legs)
    est_cost_bps = legs * float(one_way_cost_bps) + is_switch.float() * float(extra_switch_penalty_bps)
    return torch.stack(
        [
            (~is_switch).float(),
            is_switch.float(),
            (prev == int(cash_index)).float(),
            (cand == int(cash_index)).float(),
            legs,
            est_cost_bps / 100.0,
            leverage[cand] - leverage[prev],
            (groups[cand] == groups[prev]).float(),
        ],
        dim=-1,
    )


def make_constraint_features(
    *,
    bars_held: torch.Tensor,
    cooldown_remaining: torch.Tensor,
    switches_today: torch.Tensor,
    switches_episode: torch.Tensor,
    constraints: TradingConstraintConfig,
    episode_length: int,
    order_legs_today: torch.Tensor | None = None,
    order_legs_episode: torch.Tensor | None = None,
) -> torch.Tensor:
    batch = bars_held.shape[0]
    if order_legs_today is None:
        order_legs_today = torch.zeros(batch, dtype=torch.float32, device=bars_held.device)
    if order_legs_episode is None:
        order_legs_episode = torch.zeros(batch, dtype=torch.float32, device=bars_held.device)
    daily_switch_den = _cap_denominator(constraints.max_switches_per_day, episode_length)
    episode_switch_den = _cap_denominator(constraints.max_switches_per_episode, episode_length)
    daily_leg_den = _cap_denominator(constraints.max_order_legs_per_day, max(2 * episode_length, 1))
    episode_leg_den = _cap_denominator(constraints.max_order_legs_per_episode, max(2 * episode_length, 1))
    return torch.stack(
        [
            bars_held.float() / max(float(constraints.min_hold_bars), 1.0),
            cooldown_remaining.float() / max(float(constraints.cooldown_bars), 1.0),
            switches_today.float() / max(daily_switch_den, 1.0),
            switches_episode.float() / max(episode_switch_den, 1.0),
            order_legs_today.float() / max(daily_leg_den, 1.0),
            order_legs_episode.float() / max(episode_leg_den, 1.0),
        ],
        dim=1,
    ).clamp(0.0, 8.0)


def build_action_mask(
    *,
    current_action: torch.Tensor,
    bars_held: torch.Tensor,
    cooldown_remaining: torch.Tensor,
    switches_today: torch.Tensor,
    min_hold_bars: int,
    action_count: int,
    max_switches_per_day: int | None = None,
    switches_episode: torch.Tensor | None = None,
    max_switches_per_episode: int | None = None,
    order_legs_today: torch.Tensor | None = None,
    max_order_legs_per_day: float | None = None,
    order_legs_episode: torch.Tensor | None = None,
    max_order_legs_per_episode: float | None = None,
    cash_index: int = 0,
    count_etf_to_etf_as_two_legs: bool = True,
) -> torch.Tensor:
    mask = torch.ones(current_action.shape[0], action_count, dtype=torch.bool, device=current_action.device)
    must_hold = bars_held < int(min_hold_bars)
    in_cooldown = cooldown_remaining > 0
    exhausted = torch.zeros(current_action.shape[0], dtype=torch.bool, device=current_action.device)
    if max_switches_per_day is not None:
        exhausted = exhausted | (switches_today >= int(max_switches_per_day))
    if max_switches_per_episode is not None:
        if switches_episode is None:
            raise ValueError("switches_episode is required when max_switches_per_episode is set.")
        exhausted = exhausted | (switches_episode >= int(max_switches_per_episode))

    if max_order_legs_per_day is not None or max_order_legs_per_episode is not None:
        candidates = torch.arange(action_count, dtype=torch.long, device=current_action.device)
        candidates = candidates.unsqueeze(0).expand(current_action.shape[0], -1)
        previous = current_action.long().unsqueeze(1).expand_as(candidates)
        candidate_legs = trade_legs(
            previous,
            candidates,
            cash_index=cash_index,
            count_etf_to_etf_as_two_legs=count_etf_to_etf_as_two_legs,
        )
        if max_order_legs_per_day is not None:
            if order_legs_today is None:
                raise ValueError("order_legs_today is required when max_order_legs_per_day is set.")
            mask = mask & ((order_legs_today.float().unsqueeze(1) + candidate_legs) <= float(max_order_legs_per_day))
        if max_order_legs_per_episode is not None:
            if order_legs_episode is None:
                raise ValueError("order_legs_episode is required when max_order_legs_per_episode is set.")
            mask = mask & (
                (order_legs_episode.float().unsqueeze(1) + candidate_legs) <= float(max_order_legs_per_episode)
            )
        mask[torch.arange(current_action.shape[0], device=current_action.device), current_action.long()] = True

    constrained = must_hold | in_cooldown | exhausted
    if bool(constrained.any().item()):
        mask[constrained, :] = False
        mask[constrained, current_action[constrained].long()] = True
        # De-risking to CASH must never be blocked by a position-level hold (min-hold or
        # cooldown): flattening a freshly-entered, possibly leveraged position to the
        # zero-notional fallback can only reduce risk. Turnover budgets (switch/order-leg
        # caps) are intentionally NOT exempted -- a switch to cash still consumes the
        # turnover budget, so an exhausted budget may legitimately restrict to holding.
        hold_constrained = must_hold | in_cooldown
        if bool(hold_constrained.any().item()):
            mask[hold_constrained, int(cash_index)] = True
    return mask


def apply_leg_aware_hysteresis(
    q_values: torch.Tensor,
    current_action: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    one_way_cost_bps: float,
    extra_switch_penalty_bps: float,
    q_switch_margin_bps: float,
    cash_index: int = 0,
    reward_scale: float = 10_000.0,
    count_etf_to_etf_as_two_legs: bool = True,
) -> torch.Tensor:
    batch, action_count = q_values.shape
    candidates = torch.arange(action_count, dtype=torch.long, device=q_values.device).unsqueeze(0).expand(batch, -1)
    previous = current_action.long().unsqueeze(1).expand_as(candidates)
    candidate_legs = trade_legs(
        previous,
        candidates,
        cash_index=cash_index,
        count_etf_to_etf_as_two_legs=count_etf_to_etf_as_two_legs,
    )
    is_switch = candidates.ne(previous)
    required_edge = (
        candidate_legs * float(one_way_cost_bps)
        + is_switch.float() * float(extra_switch_penalty_bps)
        + is_switch.float() * float(q_switch_margin_bps)
    ) * float(reward_scale) / 10_000.0
    current_q = q_values.gather(1, current_action.long().unsqueeze(1))
    adjusted_q = q_values - current_q - required_edge
    adjusted_q = adjusted_q.masked_fill(~action_mask, torch.finfo(q_values.dtype).min)
    best_action = torch.argmax(adjusted_q, dim=1)
    best_edge = adjusted_q.gather(1, best_action.unsqueeze(1)).squeeze(1)
    should_switch = best_action.ne(current_action.long()) & (best_edge > 0)
    current_allowed = action_mask.gather(1, current_action.long().unsqueeze(1)).squeeze(1)
    return torch.where(should_switch | ~current_allowed, best_action, current_action.long())


def apply_notional_aware_hysteresis(
    q_values: torch.Tensor,
    current_action: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    action_weights: torch.Tensor,
    one_way_cost_bps: float,
    extra_switch_penalty_bps: float,
    q_switch_margin_bps: float,
    cash_index: int = 0,
    reward_scale: float = 10_000.0,
) -> torch.Tensor:
    batch, action_count = q_values.shape
    candidates = torch.arange(action_count, dtype=torch.long, device=q_values.device).unsqueeze(0).expand(batch, -1)
    previous = current_action.long().unsqueeze(1).expand_as(candidates)
    weights = action_weights.to(device=q_values.device, dtype=q_values.dtype)
    previous_weights = weights[previous]
    next_weights = weights[candidates]
    previous_weights = torch.where(previous == int(cash_index), torch.zeros_like(previous_weights), previous_weights)
    next_weights = torch.where(candidates == int(cash_index), torch.zeros_like(next_weights), next_weights)
    is_switch = candidates.ne(previous)
    traded_notional = torch.where(is_switch, previous_weights + next_weights, torch.zeros_like(next_weights))
    required_edge = (
        traded_notional * float(one_way_cost_bps)
        + is_switch.float() * float(extra_switch_penalty_bps)
        + is_switch.float() * float(q_switch_margin_bps)
    ) * float(reward_scale) / 10_000.0
    current_q = q_values.gather(1, current_action.long().unsqueeze(1))
    adjusted_q = q_values - current_q - required_edge
    adjusted_q = adjusted_q.masked_fill(~action_mask, torch.finfo(q_values.dtype).min)
    best_action = torch.argmax(adjusted_q, dim=1)
    best_edge = adjusted_q.gather(1, best_action.unsqueeze(1)).squeeze(1)
    should_switch = best_action.ne(current_action.long()) & (best_edge > 0)
    current_allowed = action_mask.gather(1, current_action.long().unsqueeze(1)).squeeze(1)
    return torch.where(should_switch | ~current_allowed, best_action, current_action.long())


def sample_valid_actions(action_mask: torch.Tensor) -> torch.Tensor:
    if not bool(action_mask.any(dim=1).all().item()):
        raise ValueError("Each action-mask row must contain at least one valid action.")
    weights = action_mask.float()
    weights = weights / weights.sum(dim=1, keepdim=True)
    return torch.multinomial(weights, num_samples=1).squeeze(1)
