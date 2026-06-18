"""Envs layer: the hour-allocation environment over sub-hour context -- state/transition/reward authority (extracted from rl_quant.minute_to_hour_transformer, protocol-first reorg Phase 4; verbatim/byte-identical, see architecture_migration_plan.md)."""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

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
from rl_quant.core import concrete_torch_device
from rl_quant.execution import (
    WeightExecutionCostConfig,
    coerce_finite_nonnegative,
    coerce_finite_positive,
    require_bool,
    require_nonnegative_int,
    require_positive_int,
    weight_transition_cost_bps,
)
from rl_quant.features.action_risk import (
    action_weight_tensor,
    build_action_metadata,
    validate_action_index_for_actions,
    validate_cash_index_for_actions,
)


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


@dataclass(frozen=True)
class TransitionCostBreakdown:
    """bps breakdown of a minute->hour transition's reward cost, SHARED by the env reward and the evaluation
    rollout so they cannot drift. The three components are deliberately distinct:
    - ``leg_cost_bps`` = ``legs * one_way_cost_bps`` -- the EXECUTION / turnover cost (what an execution-cost
      A/B should compare against);
    - ``switch_penalty_bps`` = ``is_switch * extra_switch_penalty_bps`` -- a BEHAVIOURAL anti-churn regularizer,
      NOT a market execution cost (so a cost-model swap must not silently drop it);
    - ``cash_idle_bps`` -- the idle-cash penalty when the executed action is cash.
    The legacy reward cost is ``trade_cost_bps + cash_idle_bps`` and the net return is
    ``raw_return - (trade_cost_bps + cash_idle_bps)/1e4`` (env reward = ``reward_scale * net_return``)."""

    legs: torch.Tensor
    leg_cost_bps: torch.Tensor
    switch_penalty_bps: torch.Tensor
    cash_idle_bps: torch.Tensor

    @property
    def trade_cost_bps(self) -> torch.Tensor:
        return self.leg_cost_bps + self.switch_penalty_bps


def transition_trade_cost_bps(
    previous_actions: torch.Tensor,
    actions: torch.Tensor,
    *,
    constraints: TradingConstraintConfig,
    cash_idle_penalty_bps: float,
    action_count: int | None = None,
) -> TransitionCostBreakdown:
    """Compute the shared minute->hour transition cost breakdown (see TransitionCostBreakdown). Tensor-shaped,
    so it serves both the vectorized env and a 1-element evaluation step. Pass ``action_count`` (the size of
    the action space) to additionally range-check the action indices -- env/eval do this before calling, but a
    direct caller otherwise gets a syntactically-valid but meaningless ledger from an out-of-range index."""
    # Central reward/cost accounting: reject impossible inputs (cheap metadata/scalar checks -- no device sync).
    if previous_actions.shape != actions.shape:
        raise ValueError("previous_actions and actions must have the same shape.")
    if previous_actions.device != actions.device:
        raise ValueError("previous_actions and actions must be on the same device.")
    if previous_actions.dtype not in (torch.int16, torch.int32, torch.int64) or actions.dtype not in (
        torch.int16, torch.int32, torch.int64
    ):
        raise ValueError("previous_actions and actions must be integer action-index tensors.")
    # Reuse execution.py's coercion (rejects bool, requires finite + non-negative) so a bool / NaN / inf /
    # negative / non-numeric-string bps scalar fails closed instead of silently producing a garbage cost
    # ledger. (A numeric string like "1" is still accepted and parsed, matching the execution-module config
    # contract; the guard's purpose is rejecting bool and genuinely-invalid values, not type-purity.)
    cash_idle = coerce_finite_nonnegative("cash_idle_penalty_bps", cash_idle_penalty_bps)
    one_way = coerce_finite_nonnegative("constraints.one_way_cost_bps", constraints.one_way_cost_bps)
    extra_switch = coerce_finite_nonnegative("constraints.extra_switch_penalty_bps", constraints.extra_switch_penalty_bps)
    # cash_index is compared against action indices below; reject silent bool/float/string coercion. This
    # helper has no action_names to semantically validate "is it cash" -- env/eval do that before calling --
    # but it must not let cash_index=True/0.9/"0" pick the wrong slot when called directly.
    if isinstance(constraints.cash_index, bool) or not isinstance(constraints.cash_index, Integral):
        raise ValueError(f"constraints.cash_index must be an integer, got {constraints.cash_index!r}.")
    cash_index = int(constraints.cash_index)
    # count_etf_to_etf_as_two_legs selects the leg-count branch in trade_legs; a non-bool (e.g. the string
    # "false", which is truthy) would silently pick the two-leg path, so require a real bool.
    if not isinstance(constraints.count_etf_to_etf_as_two_legs, bool):
        raise ValueError(
            "constraints.count_etf_to_etf_as_two_legs must be a bool, got "
            f"{constraints.count_etf_to_etf_as_two_legs!r}."
        )
    if action_count is not None:
        if isinstance(action_count, bool) or not isinstance(action_count, Integral):
            raise ValueError(f"action_count must be an integer, got {action_count!r}.")
        action_count = int(action_count)
        if action_count <= 0:
            raise ValueError(f"action_count must be positive, got {action_count}.")
        if not (0 <= cash_index < action_count):
            raise ValueError(f"constraints.cash_index={cash_index} is outside action_count={action_count}.")
        out_of_range = (
            (previous_actions < 0) | (previous_actions >= action_count)
            | (actions < 0) | (actions >= action_count)
        )
        if bool(out_of_range.any().item()):
            raise ValueError("previous_actions/actions contain an out-of-range action index.")
    legs = trade_legs(
        previous_actions,
        actions,
        cash_index=cash_index,
        count_etf_to_etf_as_two_legs=constraints.count_etf_to_etf_as_two_legs,
    )
    is_switch = (actions != previous_actions).float()
    return TransitionCostBreakdown(
        legs=legs,
        leg_cost_bps=legs * one_way,
        switch_penalty_bps=is_switch * extra_switch,
        cash_idle_bps=(actions == cash_index).float() * cash_idle,
    )


@dataclass(frozen=True)
class NormalizedMinuteToHourConstraints:
    """The validated, CANONICAL-typed form of the minute->hour trading constraints.

    Field names mirror TradingConstraintConfig so this is a drop-in for the attribute-reading consumers
    (transition_trade_cost_bps, make_constraint_features, build_action_mask kwargs). Every field has been
    coerced to a canonical Python type (int/float/bool, or None for an unset cap), so a numeric-string or
    numpy-scalar config value can never reach the runtime un-normalized. Frozen, so the validated values cannot
    be mutated after construction. The env/evaluator store and use THIS object instead of the raw config, which
    is what closes the "validated but still raw" drift the review flagged."""

    cash_index: int
    count_etf_to_etf_as_two_legs: bool
    one_way_cost_bps: float
    extra_switch_penalty_bps: float
    q_switch_margin_bps: float
    min_hold_bars: int
    cooldown_bars: int
    max_switches_per_day: int | None
    max_switches_per_episode: int | None
    max_order_legs_per_day: float | None
    max_order_legs_per_episode: float | None


def validate_minute_to_hour_constraints(
    constraints: TradingConstraintConfig, action_names: list[str]
) -> NormalizedMinuteToHourConstraints:
    """Validate the constraint fields that feed BOTH the action mask and the cost ledger and return their
    NORMALIZED (canonical-typed) form. Consumers should use the returned object rather than the raw config so a
    validated value is the only value used. transition_trade_cost_bps re-checks these per step, but
    build_action_mask consumes ``count_etf_to_etf_as_two_legs`` (and cash_index) BEFORE the cost helper runs --
    so a malformed value (e.g. the truthy string "false") could skew action availability / observe() first.
    Validating here, at env/eval construction, closes that ordering gap so masks never see an unvalidated (or
    un-normalized) constraint."""
    cash_index = validate_cash_index_for_actions(action_names, constraints.cash_index)
    if not isinstance(constraints.count_etf_to_etf_as_two_legs, bool):
        raise ValueError(
            "constraints.count_etf_to_etf_as_two_legs must be a bool, got "
            f"{constraints.count_etf_to_etf_as_two_legs!r}."
        )

    def _opt_int(name: str, value: object) -> int | None:  # optional cap: None = uncapped
        return None if value is None else require_nonnegative_int(name, value)

    def _opt_bps(name: str, value: object) -> float | None:
        return None if value is None else coerce_finite_nonnegative(name, value)

    # bps scalars feed the cost ledger / hysteresis scoring; reject bool/NaN/inf/negative (q_switch_margin_bps
    # NaN would poison hysteresis -- every comparison against it is False). Hold/cooldown bar counts feed the
    # mask; reject bool / fractional / negative (int(True)/int(1.9) would silently mis-gate switching).
    return NormalizedMinuteToHourConstraints(
        cash_index=cash_index,
        count_etf_to_etf_as_two_legs=bool(constraints.count_etf_to_etf_as_two_legs),
        one_way_cost_bps=coerce_finite_nonnegative("constraints.one_way_cost_bps", constraints.one_way_cost_bps),
        extra_switch_penalty_bps=coerce_finite_nonnegative(
            "constraints.extra_switch_penalty_bps", constraints.extra_switch_penalty_bps),
        q_switch_margin_bps=coerce_finite_nonnegative("constraints.q_switch_margin_bps", constraints.q_switch_margin_bps),
        min_hold_bars=require_nonnegative_int("constraints.min_hold_bars", constraints.min_hold_bars),
        cooldown_bars=require_nonnegative_int("constraints.cooldown_bars", constraints.cooldown_bars),
        max_switches_per_day=_opt_int("constraints.max_switches_per_day", constraints.max_switches_per_day),
        max_switches_per_episode=_opt_int("constraints.max_switches_per_episode", constraints.max_switches_per_episode),
        max_order_legs_per_day=_opt_bps("constraints.max_order_legs_per_day", constraints.max_order_legs_per_day),
        max_order_legs_per_episode=_opt_bps(
            "constraints.max_order_legs_per_episode", constraints.max_order_legs_per_episode),
    )


def validate_cash_usable_on_decision_rows(data: HourFromMinuteDataSplit, cash_index: int) -> None:
    """CASH is the FORCED safety fallback for masked / missing-label actions, so it must be USABLE on every
    valid decision row: its label must be VALID *and* its return FINITE. The fallback reads
    ``action_returns[row, cash_index]`` directly (a non-finite return would emit a NaN reward) and the
    usable-label rule everywhere else is ``label_valid & isfinite`` -- a finite-but-label-invalid CASH would be
    a broken safety action. The action-return contract makes label-valid <=> finite for builder-produced
    splits, but a hand-constructed split (tests/research) can violate it, so the env and evaluator both enforce
    the full usability up front and fail closed at construction/entry."""
    rows = data.valid_start_indices
    if rows.numel() == 0:
        return  # an empty split is handled by the caller's own emptiness guard
    rows = rows.to(data.action_returns.device)
    cash_returns = data.action_returns[rows, cash_index]
    cash_label_valid = data.label_valid_actions(rows)[:, cash_index]
    if not bool((cash_label_valid & torch.isfinite(cash_returns)).all().item()):
        raise ValueError(
            "CASH action must be USABLE (label-valid AND finite return) on every valid decision row "
            "(it is the forced safety fallback); the split has an unusable CASH action."
        )


class VectorizedMinuteToHourEnv:
    def __init__(self, data: HourFromMinuteDataSplit, config: MinuteToHourEnvConfig, device: torch.device) -> None:
        # initial_action gets the SHARED action-index discipline; the rest of the constraint fields that feed
        # the action mask AND the cost ledger (cash_index must be a real CASH action; count_etf must be a real
        # bool; the bps scalars finite/non-negative) are validated together at construction -- BEFORE any mask
        # is built -- so masks/observe() never see an unvalidated constraint. The NORMALIZED ints are stored
        # and used throughout env state so the constructor is the single validation point.
        self.initial_action = validate_action_index_for_actions(
            data.action_names, config.initial_action, name="initial_action"
        )
        # Store the NORMALIZED constraints (canonical types) and use them everywhere instead of the raw config,
        # so a numeric-string / numpy-scalar config value can never reach the mask / cost ledger un-normalized.
        self.constraints = validate_minute_to_hour_constraints(config.constraints, data.action_names)
        self.cash_index = self.constraints.cash_index
        # reward_scale multiplies every reward and normalises the shadow bps artifacts, so a zero/negative/
        # non-finite value would zero, flip, or blow them up -- validate and STORE the canonical float.
        self.reward_scale = coerce_finite_positive("reward_scale", config.reward_scale)
        # Sizing/penalty scalars fail closed at construction (num_envs/episode_length must be positive ints;
        # cash_idle_penalty_bps finite/non-negative). episode_length<=0 would truncate every episode at step 0.
        # Store the validated sizing ints and use them at runtime (the env never reads the raw mutable config
        # for runtime values; the economic scalars/constraints are likewise normalized + stored above/below).
        self.num_envs = require_positive_int("num_envs", config.num_envs)
        self.episode_length = require_positive_int("episode_length", config.episode_length)
        # Store the normalized (canonical float) cash-idle penalty and use it in the reward ledger.
        self.cash_idle_penalty_bps = coerce_finite_nonnegative("cash_idle_penalty_bps", config.cash_idle_penalty_bps)
        # Pin to a CONCRETE device ordinal (concrete_torch_device): the env is typically built with the result
        # of core.resolve_torch_device, and although that now returns a concrete cuda:<idx>, a caller could
        # still pass an ordinal-free torch.device("cuda"). _validate_step_actions compares an action tensor's
        # concrete device against self.device, so an ordinal-free self.device would REJECT valid CUDA actions.
        device = concrete_torch_device(device)
        self.data = data if data.minute_features.device == device else data.to(device)
        self.config = config
        # Derive self.device from the actual moved tensor so it matches what indexed tensors report exactly.
        self.device = self.data.minute_features.device
        # CASH is the forced safety fallback; require it to be USABLE (label-valid + finite) on every valid row.
        validate_cash_usable_on_decision_rows(self.data, self.cash_index)
        self.start_indices = self._build_start_index_pool()
        self.indices = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.previous_actions = torch.full((config.num_envs,), self.initial_action, dtype=torch.long, device=device)
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
        # Governed flag: require a REAL bool (bool("false") would be True -- a silent enable of the shadow).
        self.execution_env_reward_shadow = require_bool(
            "execution_env_reward_shadow", config.execution_env_reward_shadow
        )
        if self.execution_env_reward_shadow:
            self._shadow_action_weights = action_weight_tensor(
                build_action_metadata(list(self.data.action_names)), device=device
            )
            self._shadow_weight_cost = WeightExecutionCostConfig(fee_bps=self.constraints.one_way_cost_bps)
        self.reset()

    def _build_start_index_pool(self) -> torch.Tensor:
        starts = self.data.valid_start_indices
        starts = starts[starts + 1 < self.data.action_returns.shape[0]]
        if starts.numel() == 0:
            raise ValueError("No valid minute-to-hour start indices remain.")
        return starts.to(self.device)

    def reset(self, mask: torch.Tensor | None = None) -> None:
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        count = int(mask.sum().item())
        if count == 0:
            return
        random_ids = torch.randint(0, self.start_indices.shape[0], (count,), device=self.device)
        self.indices[mask] = self.start_indices[random_ids]
        self.previous_actions[mask] = self.initial_action
        self.bars_held[mask] = int(self.constraints.min_hold_bars)
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
            constraints=self.constraints,
            episode_length=self.episode_length,
            order_legs_today=self.order_legs_today,
            order_legs_episode=self.order_legs_episode,
        )

    def action_mask(self, row_indices: torch.Tensor | None = None) -> torch.Tensor:
        constraint_mask = build_action_mask(
            current_action=self.previous_actions,
            bars_held=self.bars_held,
            cooldown_remaining=self.cooldown_remaining,
            switches_today=self.switches_today,
            max_switches_per_day=self.constraints.max_switches_per_day,
            min_hold_bars=self.constraints.min_hold_bars,
            action_count=len(self.data.action_names),
            switches_episode=self.switches_episode,
            max_switches_per_episode=self.constraints.max_switches_per_episode,
            order_legs_today=self.order_legs_today,
            max_order_legs_per_day=self.constraints.max_order_legs_per_day,
            order_legs_episode=self.order_legs_episode,
            max_order_legs_per_episode=self.constraints.max_order_legs_per_episode,
            cash_index=self.cash_index,
            count_etf_to_etf_as_two_legs=self.constraints.count_etf_to_etf_as_two_legs,
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
        availability_mask[:, self.cash_index] = True
        mask = constraint_mask & availability_mask
        empty_rows = ~mask.any(dim=1)
        if bool(empty_rows.any().item()):
            mask[empty_rows, self.cash_index] = True
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

    def _validate_step_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Fail closed on a malformed action tensor at the env boundary, before it reaches ``gather``.

        ``.long()`` would silently truncate a float (2.9 -> 2) or coerce a bool; a wrong device/shape or an
        out-of-range index would otherwise raise an opaque PyTorch/CUDA error (or, for an in-range-but-wrong
        shape, mis-step). The fallback logic only handles VALID-but-masked actions, not malformed input."""
        if actions.dtype not in (torch.int16, torch.int32, torch.int64):
            raise ValueError(f"actions must be an integer action-index tensor, got dtype {actions.dtype}.")
        if actions.device != self.device:
            raise ValueError(f"actions must be on device {self.device}, got {actions.device}.")
        if actions.shape != (self.num_envs,):
            raise ValueError(
                f"actions must have shape ({self.num_envs},), got {tuple(actions.shape)}."
            )
        actions = actions.long()
        if bool(((actions < 0) | (actions >= len(self.data.action_names))).any().item()):
            raise ValueError("actions contain an out-of-range action index.")
        return actions

    def step(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        current_indices = self.indices.clone()
        previous_actions = self.previous_actions.clone()
        # PR-D: snapshot the dynamic state of the position held ENTERING this decision (before the update
        # below). Always computed and returned (the replay add() filters unknown keys, so it is harmless when
        # use_dynamic_transition_features is off); consumed only when the flag is on.
        position_dynamic = self.dynamic_state()
        constraint_features = self.constraint_features()
        action_mask = self.action_mask()
        actions = self._validate_step_actions(actions)
        selected_valid = action_mask.gather(1, actions.unsqueeze(1)).squeeze(1)
        # On an invalid requested action, de-risk to CASH when it is valid (a masked action should fall back to
        # cash, not to whatever happens to be the first valid column -- argmax alone only yields CASH if CASH is
        # index 0). Falls back to the first valid action only when CASH itself is masked. For the canonical
        # cash_index==0 universe (CASH always valid via the data-quality fallback) this is byte-identical to the
        # old argmax (both pick index 0), and it removes the dependency on CASH being the first action.
        first_valid_actions = torch.argmax(action_mask.long(), dim=1)
        cash_is_valid = action_mask[:, self.cash_index]
        fallback_actions = torch.where(
            cash_is_valid, torch.full_like(actions, self.cash_index), first_valid_actions
        )
        actions = torch.where(selected_valid, actions, fallback_actions)
        # A label is only USABLE if the mask says valid AND the return is finite -- otherwise the env would
        # train on a NaN/inf reward. This matches evaluate_minute_to_hour_policy (which already checks both),
        # so the env and the evaluator fall back to CASH on exactly the same rows. For clean data (label-valid
        # implies finite, per the protocol's NaN-marks-invalid contract) this is byte-identical.
        label_mask = self.data.label_valid_actions(current_indices)
        usable_labels = label_mask & torch.isfinite(self.data.action_returns[current_indices])
        selected_label_valid = usable_labels.gather(1, actions.unsqueeze(1)).squeeze(1)
        cash_actions = torch.full_like(actions, self.cash_index)
        actions = torch.where(selected_label_valid, actions, cash_actions)
        raw_returns = self.data.action_returns[current_indices, actions]
        is_switch = actions != previous_actions
        cost = transition_trade_cost_bps(
            previous_actions,
            actions,
            constraints=self.constraints,
            cash_idle_penalty_bps=self.cash_idle_penalty_bps,
            action_count=len(self.data.action_names),
        )
        legs = cost.legs
        cost_bps = cost.trade_cost_bps  # leg cost + switch penalty (the legacy combined trade cost)
        cash_idle_penalty_bps = cost.cash_idle_bps
        rewards = raw_returns * self.reward_scale - (
            cost_bps + cash_idle_penalty_bps
        ) * self.reward_scale / 10_000.0
        # PR-3 shadow (computed from the SAME state, only logged -- `rewards` above, used for training, is
        # untouched, so this is byte-identical to shadow-off). Weight-bps cost of the transition's two legs
        # (sell prior weight + buy new weight; cash = no exposure; only a real switch trades), carrying the same
        # cash-idle term as the legacy reward so reward_delta isolates the trade-cost-MODEL change.
        if self.execution_env_reward_shadow:
            cash_idx = self.cash_index
            w_prev = self._shadow_action_weights[previous_actions]
            w_next = self._shadow_action_weights[actions]
            zeros = torch.zeros_like(w_prev)
            sell_weight = torch.where(is_switch & (previous_actions != cash_idx), w_prev, zeros)
            buy_weight = torch.where(is_switch & (actions != cash_idx), w_next, zeros)
            execution_cost_bps_shadow = weight_transition_cost_bps(
                sell_weight, buy_weight, weight_cost=self._shadow_weight_cost
            )
            # Swap ONLY the execution/leg cost (cost.leg_cost_bps -> execution_cost_bps_shadow); KEEP the
            # behavioural switch-penalty regularizer + the cash-idle penalty, so reward_delta / cost_delta
            # isolate the cost-MODEL change and PR-4 would not silently drop the anti-churn regularizer.
            execution_env_reward_shadow = raw_returns * self.reward_scale - (
                execution_cost_bps_shadow + cost.switch_penalty_bps + cash_idle_penalty_bps
            ) * self.reward_scale / 10_000.0

        next_indices = current_indices + 1
        self.indices = next_indices
        self.previous_actions = actions
        self.bars_held = torch.where(is_switch, torch.ones_like(self.bars_held), self.bars_held + 1)
        self.cooldown_remaining = torch.where(
            is_switch,
            torch.full_like(self.cooldown_remaining, int(self.constraints.cooldown_bars)),
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
        truncated = self.steps >= self.episode_length
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
        # training -- they are for logging / the shadow A/B only.
        if self.execution_env_reward_shadow:
            out["execution_env_reward_shadow"] = execution_env_reward_shadow
            out["execution_cost_bps_shadow"] = execution_cost_bps_shadow
            out["reward_delta_shadow"] = execution_env_reward_shadow - rewards
            # vs the legacy LEG/execution cost (excludes the switch-penalty regularizer, which both rewards
            # keep) -> a pure execution-cost-model delta.
            out["cost_delta_shadow"] = execution_cost_bps_shadow - cost.leg_cost_bps
        return out
