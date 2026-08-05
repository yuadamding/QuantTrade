"""Authoritative delayed-fill runtime for the Hold-30 mechanism screen.

The runtime deliberately separates the three economically different books in
the Hold-30 RFC:

``decision_weights[t]``
    The post-cost book visible to the actor at decision ``t``.
``execution_pretrade_weights[t + 1]``
    The old book after its ``t -> t+1`` return and cohort-age advance.
``repaired_weights[t + 1]``
    The fill-time book after membership, availability, and risk repairs.

An intent is emitted exactly once at ``t`` and is retained in a typed pending
queue.  It cannot see any return, membership, availability, or risk input from
``t+1``.  The pending intent is constructed and filled only after the old book
has earned its return and the mandatory repairs have been applied.  The net
utility row indexed by ``t`` therefore contains the old-book return and all
fill costs at ``t+1``; the newly filled book earns its first return on row
``t+1``.

The same transition function drives both the canonical no-grad chronology and
the differentiable origin replay.  A replay restores a detached canonical
boundary, attaches only the origin intent, and uses detached canonical intents
for the following thirty support decisions.  This is the package-owned
implementation of :class:`rl_quant.training.hold30.Hold30ReplayAdapter`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import torch

from rl_quant.envs.hold30 import (
    TURNOVER_CAUSES,
    CohortLedger,
    CohortTradeAccounting,
    TurnoverCause,
)
from rl_quant.execution.hold30 import (
    HOLD30_MAX_STOCK_WEIGHT,
    Hold30BuiltAction,
    build_alpha_hold30_action,
    build_h2_hold30_action,
    build_scalar_gate_hold30_action,
)
from rl_quant.execution.hold30_sleeves import (
    Hold30SleeveRepair,
    Hold30SleeveReview,
    Hold30SleeveSnapshot,
    Hold30SleeveState,
)
from rl_quant.models.daily_policy import Hold30Intent
from rl_quant.protocol.hold30_alpha_v3 import HOLD30_ALPHA_HORIZONS
from rl_quant.training.hold30 import (
    Hold30CanonicalRow,
    Hold30CreditRoles,
    Hold30OriginReplay,
    benchmark_relative_log_utility,
)

HOLD30_RECONCILIATION_TOLERANCE = 1e-6


class Hold30SafetyProjectionError(RuntimeError):
    """Raised when the action builder did not construct a feasible fill."""


def _check_floating(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {getattr(value, 'shape', None)}")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be a finite floating-point tensor")


def _check_mask(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != torch.bool:
        raise ValueError(f"{name} must be a boolean tensor with shape {shape}")


def _clone_intent(intent: Hold30Intent, *, detach: bool) -> Hold30Intent:
    if not isinstance(intent, Hold30Intent):
        raise TypeError("the policy must return Hold30Intent")

    def copy(value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError("every populated Hold30Intent field must be a floating tensor")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Hold30Intent contains a non-finite value")
        return value.detach().clone() if detach else value

    return Hold30Intent(
        entry_scores=copy(intent.entry_scores),
        target_logits=copy(intent.target_logits),
        gate=copy(intent.gate),
        hazard_residual=copy(intent.hazard_residual),
        raw_hazard_residual=copy(intent.raw_hazard_residual),
        exact_hold_probability=copy(intent.exact_hold_probability),
        exact_hold_logit=copy(intent.exact_hold_logit),
        exact_hold_soft_probability=copy(intent.exact_hold_soft_probability),
        exact_hold_decision_st=copy(intent.exact_hold_decision_st),
        exposure_residual=copy(intent.exposure_residual),
        alpha_mean_30d=copy(intent.alpha_mean_30d),
        alpha_downside_30d=copy(intent.alpha_downside_30d),
        active_risk_scale=copy(intent.active_risk_scale),
        signal_confidence=copy(intent.signal_confidence),
        uncalibrated_signal_confidence_logit=copy(
            intent.uncalibrated_signal_confidence_logit
        ),
        benchmark_derisk_request=copy(intent.benchmark_derisk_request),
        total_risk_overlay=copy(intent.total_risk_overlay),
        auxiliary_alpha_mean=copy(intent.auxiliary_alpha_mean),
    )


def _intent_tensors(intent: Hold30Intent) -> tuple[tuple[str, torch.Tensor], ...]:
    return tuple(
        (name, value)
        for name in (
            "entry_scores",
            "target_logits",
            "gate",
            "hazard_residual",
            "raw_hazard_residual",
            "exact_hold_probability",
            "exact_hold_logit",
            "exact_hold_soft_probability",
            "exact_hold_decision_st",
            "exposure_residual",
            "alpha_mean_30d",
            "alpha_downside_30d",
            "active_risk_scale",
            "signal_confidence",
            "uncalibrated_signal_confidence_logit",
            "benchmark_derisk_request",
            "total_risk_overlay",
            "auxiliary_alpha_mean",
        )
        if (value := getattr(intent, name)) is not None
    )


def _detach_sleeve_snapshot(value: Hold30SleeveSnapshot) -> Hold30SleeveSnapshot:
    return Hold30SleeveSnapshot(
        books=value.books.detach().clone(),
        session_index=value.session_index,
        last_review_session=value.last_review_session.detach().clone(),
        review_count=value.review_count.detach().clone(),
        cash_index=value.cash_index,
    )


@dataclass(frozen=True)
class Hold30Sequence:
    """Immutable tensors for one chronological Hold-30 sweep.

    ``n_positions`` includes the final state-only terminal observation.  All
    fill-time tensors are indexed by position, while ``asset_returns`` and
    ``benchmark_net_returns`` are indexed by the originating decision row and
    consequently contain ``n_positions - 1`` rows.

    ``decision_state`` is already a causal policy representation.  The runtime
    passes only its row, the decision-time mask, the decision book, and age
    summaries to the actor.  It never passes a fill-time tensor to the actor.
    """

    decision_state: torch.Tensor
    asset_returns: torch.Tensor
    decision_available: torch.Tensor
    fill_membership: torch.Tensor
    fill_availability: torch.Tensor
    benchmark_weights: torch.Tensor
    risk_asset_caps: torch.Tensor
    risk_gross_max: torch.Tensor
    benchmark_net_returns: torch.Tensor
    initial_ledger: CohortLedger
    cost_rate: float | torch.Tensor = 0.002
    initial_equity: torch.Tensor | None = None
    track_entry_units: torch.Tensor | None = None
    axis_id: str = "synthetic-axis"

    def __post_init__(self) -> None:
        if not isinstance(self.axis_id, str) or not self.axis_id:
            raise ValueError("axis_id must be a non-empty string")
        if not isinstance(self.decision_state, torch.Tensor) or self.decision_state.ndim < 4:
            raise ValueError("decision_state must have shape [position, batch, asset, ...]")
        positions, batch, assets = self.decision_state.shape[:3]
        if positions < 2:
            raise ValueError("a Hold-30 sequence needs at least one decision and one terminal state")
        numeric_position_shape = (positions, batch, assets)
        numeric_return_shape = (positions - 1, batch, assets)
        _check_floating("asset_returns", self.asset_returns, numeric_return_shape)
        _check_mask("decision_available", self.decision_available, numeric_position_shape)
        _check_mask("fill_membership", self.fill_membership, numeric_position_shape)
        _check_mask("fill_availability", self.fill_availability, numeric_position_shape)
        _check_floating("benchmark_weights", self.benchmark_weights, numeric_position_shape)
        _check_floating("risk_asset_caps", self.risk_asset_caps, numeric_position_shape)
        _check_floating("risk_gross_max", self.risk_gross_max, (positions, batch))
        _check_floating("benchmark_net_returns", self.benchmark_net_returns, (positions - 1, batch))
        if self.initial_ledger.batch_size != batch or self.initial_ledger.num_assets != assets:
            raise ValueError("initial_ledger does not match sequence batch and asset axes")
        reference = self.asset_returns
        for name, value in (
            ("benchmark_weights", self.benchmark_weights),
            ("risk_asset_caps", self.risk_asset_caps),
            ("risk_gross_max", self.risk_gross_max),
            ("benchmark_net_returns", self.benchmark_net_returns),
        ):
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"{name} must share the return dtype and device")
        if (
            self.initial_ledger.economic_value.device != reference.device
            or self.initial_ledger.economic_value.dtype != reference.dtype
        ):
            raise ValueError("initial_ledger must share the return dtype and device")
        cash = self.initial_ledger.cash_index
        for name, mask in (
            ("decision_available", self.decision_available),
            ("fill_membership", self.fill_membership),
            ("fill_availability", self.fill_availability),
        ):
            if not bool(mask[..., cash].all()):
                raise ValueError(f"CASH must always be true in {name}")
        if bool((self.asset_returns <= -1.0).any()):
            raise ValueError("asset returns must be greater than -1")
        if bool((self.benchmark_net_returns <= -1.0).any()):
            raise ValueError("benchmark net returns must be greater than -1")
        if bool((self.risk_asset_caps < 0).any()) or bool((self.risk_gross_max < 0).any()):
            raise ValueError("risk caps cannot be negative")
        benchmark_totals = self.benchmark_weights.sum(-1)
        if bool((self.benchmark_weights < -HOLD30_RECONCILIATION_TOLERANCE).any()) or not bool(
            torch.allclose(benchmark_totals, torch.ones_like(benchmark_totals), atol=1e-6, rtol=1e-6)
        ):
            raise ValueError("benchmark_weights must be nonnegative simplexes")
        self.initial_ledger.assert_reconciles(self.initial_ledger.weights)
        if self.initial_equity is not None:
            _check_floating("initial_equity", self.initial_equity, (batch,))
            if self.initial_equity.device != reference.device or self.initial_equity.dtype != reference.dtype:
                raise ValueError("initial_equity must share the return dtype and device")
            if bool((self.initial_equity <= 0).any()):
                raise ValueError("initial_equity must be positive")
        rates = self.cost_rates
        if not bool(torch.isfinite(rates).all()) or bool((rates < 0).any()):
            raise ValueError("cost_rate must be finite and nonnegative")
        if self.track_entry_units is not None and (
            not isinstance(self.track_entry_units, torch.Tensor)
            or self.track_entry_units.dtype != torch.bool
            or tuple(self.track_entry_units.shape) != (positions - 1,)
        ):
            raise ValueError(
                "track_entry_units must be a common boolean [decision] mask; "
                "split sequences when batch rows have different score/support roles"
            )

    @property
    def n_positions(self) -> int:
        return int(self.decision_state.shape[0])

    @property
    def batch_size(self) -> int:
        return int(self.decision_state.shape[1])

    @property
    def num_assets(self) -> int:
        return int(self.decision_state.shape[2])

    @property
    def cash_index(self) -> int:
        return self.initial_ledger.cash_index

    @property
    def cost_rates(self) -> torch.Tensor:
        value = torch.as_tensor(
            self.cost_rate,
            dtype=self.asset_returns.dtype,
            device=self.asset_returns.device,
        )
        try:
            return torch.broadcast_to(value, (self.n_positions - 1, self.batch_size))
        except RuntimeError as exc:
            raise ValueError("cost_rate must broadcast to [decision, batch]") from exc

    @property
    def entry_tracking_mask(self) -> torch.Tensor:
        if self.track_entry_units is None:
            return torch.ones(
                self.n_positions - 1,
                dtype=torch.bool,
                device=self.asset_returns.device,
            )
        return self.track_entry_units.to(device=self.asset_returns.device)


@dataclass(frozen=True)
class PendingHold30Intent:
    """Receipt-complete delayed instruction waiting for its legal fill."""

    intent: Hold30Intent
    decision_index: int
    fill_index: int
    decision_available: torch.Tensor
    axis_id: str

    def detach(self) -> PendingHold30Intent:
        return PendingHold30Intent(
            intent=_clone_intent(self.intent, detach=True),
            decision_index=self.decision_index,
            fill_index=self.fill_index,
            decision_available=self.decision_available.detach().clone(),
            axis_id=self.axis_id,
        )


@dataclass(frozen=True)
class Hold30RuntimeState:
    """Economic and delayed-fill state required for exact continuation."""

    position_index: int
    ledger: CohortLedger
    equity: torch.Tensor
    pending_intent: PendingHold30Intent | None = None
    sleeve_snapshot: Hold30SleeveSnapshot | None = None

    def detach(self) -> Hold30RuntimeState:
        return Hold30RuntimeState(
            position_index=self.position_index,
            ledger=self.ledger.detach().clone(),
            equity=self.equity.detach().clone(),
            pending_intent=None if self.pending_intent is None else self.pending_intent.detach(),
            sleeve_snapshot=(
                None
                if self.sleeve_snapshot is None
                else _detach_sleeve_snapshot(self.sleeve_snapshot)
            ),
        )


@dataclass(frozen=True)
class Hold30Transition:
    """Every action and economic stage for one delayed-fill transition."""

    decision_index: int
    fill_index: int
    raw_intent: Hold30Intent
    decision_weights: torch.Tensor
    execution_pretrade_weights: torch.Tensor
    retention_units_before_membership: torch.Tensor
    membership_repaired_weights: torch.Tensor
    availability_repaired_weights: torch.Tensor
    risk_repaired_weights: torch.Tensor
    retention_units_after_forced: torch.Tensor
    requested_delta: torch.Tensor
    constructed_delta: torch.Tensor
    filled_delta: torch.Tensor
    pre_cost_weights: torch.Tensor
    post_cost_weights: torch.Tensor
    holding_return: torch.Tensor
    cost: torch.Tensor
    cost_financing: torch.Tensor
    net_return: torch.Tensor
    benchmark_net_return: torch.Tensor
    utility: torch.Tensor
    projection_distance: torch.Tensor
    turnover_by_cause: dict[TurnoverCause, torch.Tensor]
    accounting_by_cause: dict[TurnoverCause, CohortTradeAccounting]
    discretionary_accounting: CohortTradeAccounting
    sleeve_repair: Hold30SleeveRepair | None
    sleeve_review: Hold30SleeveReview | None
    equity_before: torch.Tensor
    equity_after: torch.Tensor


@dataclass(frozen=True)
class Hold30CanonicalTrace:
    """Detached replay boundaries and deployed actions from Pass A."""

    boundary_states: tuple[Hold30RuntimeState, ...]
    decision_states: tuple[torch.Tensor, ...]
    pending_intents: tuple[PendingHold30Intent, ...]
    transitions: tuple[Hold30Transition, ...]

    @property
    def terminal_state(self) -> Hold30RuntimeState:
        return self.boundary_states[-1]


class Hold30Policy(Protocol):
    def hold30_intent(
        self,
        state_t: torch.Tensor,
        prev_weights: torch.Tensor,
        available: torch.Tensor,
        age_summaries: torch.Tensor | None = None,
    ) -> Hold30Intent: ...


class Hold30DecisionStateProvider(Protocol):
    """Recompute causal actor state without granting fill-time information.

    A trainable provider must run the encoder/temporal path under the current
    policy parameters in both paths.  ``canonical_states`` is called once
    under ``no_grad``.  ``replay_origin_states`` is called once per feasible
    origin batch with autograd enabled; support decisions reuse their detached
    canonical raw intents and never recompute a state.
    """

    trains_upstream_encoder: bool

    def canonical_states(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
    ) -> Sequence[torch.Tensor] | torch.Tensor: ...

    def replay_origin_state(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
        origin: int,
    ) -> torch.Tensor: ...

    def replay_origin_states(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
        origins: torch.Tensor,
    ) -> Sequence[torch.Tensor] | torch.Tensor: ...


@dataclass(frozen=True)
class TensorHold30DecisionStateProvider:
    """Static state fixture/evaluation provider; it cannot train an encoder."""

    trains_upstream_encoder: bool = False

    def canonical_states(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
    ) -> torch.Tensor:
        del policy
        return sequence.decision_state[:-1]

    def replay_origin_state(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
        origin: int,
    ) -> torch.Tensor:
        del policy
        return sequence.decision_state[origin]

    def replay_origin_states(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
        origins: torch.Tensor,
    ) -> torch.Tensor:
        del policy
        index = torch.as_tensor(origins, dtype=torch.long, device=sequence.decision_state.device)
        return sequence.decision_state.index_select(0, index)


@dataclass(frozen=True)
class FunctionalHold30DecisionStateProvider:
    """Package-level callback adapter for a differentiable causal encoder.

    The callbacks may run an in-memory, streaming, or cache-recomputed encoder,
    but both must implement the same legal causal state.  Runtime numeric
    equality checks fail closed if the origin recomputation differs from the
    canonical state.  This adapter is intentionally explicit: merely passing a
    detached ``Hold30Sequence.decision_state`` never claims to train the raw or
    temporal encoder.
    """

    canonical_fn: Callable[[Hold30Policy, Hold30Sequence], Sequence[torch.Tensor] | torch.Tensor]
    replay_origin_fn: Callable[[Hold30Policy, Hold30Sequence, int], torch.Tensor]
    replay_origin_states_fn: (
        Callable[
            [Hold30Policy, Hold30Sequence, torch.Tensor],
            Sequence[torch.Tensor] | torch.Tensor,
        ]
        | None
    ) = None
    trains_upstream_encoder: bool = True

    def canonical_states(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
    ) -> Sequence[torch.Tensor] | torch.Tensor:
        return self.canonical_fn(policy, sequence)

    def replay_origin_state(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
        origin: int,
    ) -> torch.Tensor:
        return self.replay_origin_fn(policy, sequence, origin)

    def replay_origin_states(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
        origins: torch.Tensor,
    ) -> Sequence[torch.Tensor] | torch.Tensor:
        if self.replay_origin_states_fn is not None:
            return self.replay_origin_states_fn(policy, sequence, origins)
        return torch.stack(
            [
                self.replay_origin_fn(policy, sequence, int(origin))
                for origin in torch.as_tensor(origins, dtype=torch.long, device="cpu").tolist()
            ]
        )


class Hold30ActionBuilder(Protocol):
    def __call__(
        self,
        intent: Hold30Intent,
        repaired_ledger: CohortLedger,
        benchmark_weights: torch.Tensor,
        trade_mask: torch.Tensor,
        risk_asset_caps: torch.Tensor,
        risk_gross_max: torch.Tensor,
    ) -> Hold30BuiltAction: ...


def _zero_accounting(ledger: CohortLedger, cause: TurnoverCause) -> CohortTradeAccounting:
    return ledger.trade_to(ledger.weights, cause=cause)[1]


def _force_mask_to_cash(weights: torch.Tensor, allowed: torch.Tensor, cash_index: int) -> torch.Tensor:
    target = torch.where(allowed, weights, torch.zeros_like(weights))
    target = target.clone()
    risky = torch.ones_like(target, dtype=torch.bool)
    risky[:, cash_index] = False
    target[:, cash_index] = 1.0 - torch.where(risky, target, torch.zeros_like(target)).sum(-1)
    return target


def _risk_project(
    weights: torch.Tensor,
    asset_caps: torch.Tensor,
    gross_max: torch.Tensor,
    cash_index: int,
) -> torch.Tensor:
    risky = torch.ones_like(weights, dtype=torch.bool)
    risky[:, cash_index] = False
    cap = torch.where(
        risky,
        torch.minimum(asset_caps.clamp_min(0.0), weights.new_tensor(HOLD30_MAX_STOCK_WEIGHT)),
        torch.zeros_like(asset_caps),
    )
    held = torch.where(risky, weights.clamp_min(0.0), torch.zeros_like(weights))
    held = torch.minimum(held, cap)
    hard_gross = torch.minimum(
        torch.ones_like(gross_max),
        torch.minimum(gross_max.clamp_min(0.0), cap.sum(-1)),
    )
    gross = held.sum(-1)
    scale = torch.where(gross > hard_gross, hard_gross / gross.clamp_min(1e-18), torch.ones_like(gross))
    held = held * scale.unsqueeze(-1)
    target = held.clone()
    target[:, cash_index] = 1.0 - held.sum(-1)
    return target


def _one_way(new: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
    return 0.5 * (new - old).abs().sum(-1)


def _built_from_sleeve_review(
    sleeve_state: Hold30SleeveState,
    repaired_weights: torch.Tensor,
    review: Hold30SleeveReview,
) -> Hold30BuiltAction:
    target = sleeve_state.aggregate_weights()
    constructed = target - repaired_weights
    if not bool(torch.allclose(constructed, review.constructed_delta, atol=1e-6, rtol=1e-6)):
        raise RuntimeError("H3 sleeve review does not reconcile to its aggregate delta")
    risky = torch.ones_like(target, dtype=torch.bool)
    risky[:, sleeve_state.cash_index] = False
    return Hold30BuiltAction(
        target_weights=target,
        requested_delta=review.requested_delta,
        constructed_delta=constructed,
        requested_turnover=review.requested_turnover,
        constructed_turnover=_one_way(target, repaired_weights),
        desired_risky_exposure=torch.where(
            risky, target, torch.zeros_like(target)
        ).sum(-1),
        proposed_release_by_age=target.new_zeros((*target.shape, 61)),
        proposed_release=torch.zeros_like(target),
        capacity_shortfall=review.unallocated_sleeve_cash,
    )


class Hold30ChronologicalRuntime:
    """Single source of truth for canonical and differentiable transitions."""

    def __init__(
        self,
        mechanism: str,
        *,
        action_builder: Hold30ActionBuilder | None = None,
        state_provider: Hold30DecisionStateProvider | None = None,
        require_trainable_state_provider: bool = False,
        reconciliation_tolerance: float = HOLD30_RECONCILIATION_TOLERANCE,
        alpha_total_risk_step: float | None = None,
    ) -> None:
        if mechanism not in {"H0", "H1", "H2", "H3"}:
            raise ValueError("mechanism must be H0, H1, H2, or H3")
        if not 0 < float(reconciliation_tolerance) <= 1e-3:
            raise ValueError("reconciliation_tolerance must lie in (0, 1e-3]")
        if mechanism == "H3" and action_builder is not None:
            raise ValueError("H3 uses the package-owned sleeve builder and cannot replace it")
        provider = TensorHold30DecisionStateProvider() if state_provider is None else state_provider
        if not isinstance(getattr(provider, "trains_upstream_encoder", None), bool):
            raise TypeError("state_provider must declare boolean trains_upstream_encoder")
        for method in ("canonical_states", "replay_origin_states"):
            if not callable(getattr(provider, method, None)):
                raise TypeError(f"state_provider must implement callable {method}")
        if require_trainable_state_provider and not provider.trains_upstream_encoder:
            raise ValueError(
                "training the Hold-30 actor path requires a differentiable decision-state provider; "
                "a precomputed decision_state cannot train the upstream encoder"
            )
        self.mechanism = mechanism
        self.action_builder = action_builder
        self.state_provider = provider
        self.require_trainable_state_provider = bool(require_trainable_state_provider)
        self.reconciliation_tolerance = float(reconciliation_tolerance)
        if alpha_total_risk_step is not None and (
            isinstance(alpha_total_risk_step, bool)
            or not isinstance(alpha_total_risk_step, (int, float))
            or not torch.isfinite(torch.tensor(float(alpha_total_risk_step)))
            or float(alpha_total_risk_step) < 0
        ):
            raise ValueError("alpha_total_risk_step must be finite non-negative or None")
        self.alpha_total_risk_step = (
            None if alpha_total_risk_step is None else float(alpha_total_risk_step)
        )

    def initial_state(self, sequence: Hold30Sequence) -> Hold30RuntimeState:
        equity = (
            torch.ones(
                sequence.batch_size,
                dtype=sequence.asset_returns.dtype,
                device=sequence.asset_returns.device,
            )
            if sequence.initial_equity is None
            else sequence.initial_equity.clone()
        )
        sleeve_snapshot = (
            Hold30SleeveState.from_portfolio(
                sequence.initial_ledger.weights,
                cash_index=sequence.cash_index,
            ).capture()
            if self.mechanism == "H3"
            else None
        )
        return Hold30RuntimeState(
            position_index=0,
            ledger=sequence.initial_ledger.clone(),
            equity=equity,
            pending_intent=None,
            sleeve_snapshot=sleeve_snapshot,
        )

    def decide(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
        state: Hold30RuntimeState,
        *,
        decision_state: torch.Tensor | None = None,
    ) -> Hold30RuntimeState:
        """Evaluate the actor once using decision-visible inputs only."""

        self._validate_state(sequence, state)
        if state.pending_intent is not None:
            raise RuntimeError("a pending intent must fill before another decision")
        t = state.position_index
        if t >= sequence.n_positions - 1:
            raise RuntimeError("the terminal position has no decision")
        if decision_state is None:
            if not isinstance(self.state_provider, TensorHold30DecisionStateProvider):
                raise RuntimeError(
                    "a functional state provider requires an explicitly recomputed decision_state"
                )
            decision_state = sequence.decision_state[t]
        self._validate_decision_state(decision_state, sequence)
        intent = policy.hold30_intent(
            decision_state,
            state.ledger.weights,
            sequence.decision_available[t],
            state.ledger.age_summaries(),
        )
        intent = _clone_intent(intent, detach=False)
        self._validate_intent(intent, sequence.batch_size, sequence.num_assets)
        pending = PendingHold30Intent(
            intent=intent,
            decision_index=t,
            fill_index=t + 1,
            decision_available=sequence.decision_available[t].clone(),
            axis_id=sequence.axis_id,
        )
        return replace(state, pending_intent=pending)

    def attach_pending(
        self,
        sequence: Hold30Sequence,
        state: Hold30RuntimeState,
        pending: PendingHold30Intent,
    ) -> Hold30RuntimeState:
        """Attach a canonical support intent without evaluating the actor."""

        self._validate_state(sequence, state)
        if state.pending_intent is not None:
            raise RuntimeError("state already contains a pending intent")
        if pending.axis_id != sequence.axis_id:
            raise ValueError("pending intent axis_id does not match sequence")
        if pending.decision_index != state.position_index or pending.fill_index != state.position_index + 1:
            raise ValueError("pending intent indices do not match the runtime boundary")
        if not torch.equal(pending.decision_available, sequence.decision_available[state.position_index]):
            raise ValueError("pending intent decision mask does not match the bound sequence")
        self._validate_intent(pending.intent, sequence.batch_size, sequence.num_assets)
        return replace(state, pending_intent=pending)

    def advance(
        self,
        sequence: Hold30Sequence,
        state: Hold30RuntimeState,
    ) -> tuple[Hold30RuntimeState, Hold30Transition]:
        """Earn the old-book return, repair, fill the pending intent, and cost once."""

        self._validate_state(sequence, state)
        pending = state.pending_intent
        if pending is None:
            raise RuntimeError("advance requires one pending intent")
        t = state.position_index
        if pending.axis_id != sequence.axis_id or pending.decision_index != t or pending.fill_index != t + 1:
            raise ValueError("pending-intent receipt does not match this transition")

        decision_ledger = state.ledger
        decision_weights = decision_ledger.weights
        period_returns = sequence.asset_returns[t]
        holding_return = (decision_weights * period_returns).sum(-1)
        execution_pretrade = decision_ledger.age_and_drift(period_returns)
        execution_pretrade_weights = execution_pretrade.weights

        fill = t + 1
        accounting: dict[TurnoverCause, CohortTradeAccounting] = {}
        sleeve_repair: Hold30SleeveRepair | None = None
        sleeve_review: Hold30SleeveReview | None = None
        sleeve_state: Hold30SleeveState | None = None
        if self.mechanism == "H3":
            assert state.sleeve_snapshot is not None
            sleeve_state = Hold30SleeveState.from_snapshot(state.sleeve_snapshot)
            sleeve_growth = sleeve_state.drift_(period_returns)
            self._assert_close("H3 drift growth", sleeve_growth - 1.0, holding_return)
            self._assert_close(
                "H3 execution-pretrade book",
                sleeve_state.aggregate_weights(),
                execution_pretrade_weights,
            )
            sleeve_repair = sleeve_state.apply_forced_repairs_(
                sequence.fill_membership[fill],
                sequence.fill_availability[fill],
                sequence.risk_asset_caps[fill],
                sequence.risk_gross_max[fill],
            )
            membership_target = (
                execution_pretrade_weights
                + sleeve_repair.membership_forced_delta.sum(dim=1)
            )
        else:
            membership_target = _force_mask_to_cash(
                execution_pretrade_weights,
                sequence.fill_membership[fill],
                sequence.cash_index,
            )
        membership_ledger, accounting[TurnoverCause.MEMBERSHIP_FORCED] = execution_pretrade.trade_to(
            membership_target, cause=TurnoverCause.MEMBERSHIP_FORCED
        )
        availability_target = (
            membership_ledger.weights
            + sleeve_repair.availability_forced_delta.sum(dim=1)
            if sleeve_repair is not None
            else _force_mask_to_cash(
                membership_ledger.weights,
                sequence.fill_availability[fill],
                sequence.cash_index,
            )
        )
        availability_ledger, accounting[TurnoverCause.AVAILABILITY_FORCED] = membership_ledger.trade_to(
            availability_target, cause=TurnoverCause.AVAILABILITY_FORCED
        )
        risk_target = (
            availability_ledger.weights + sleeve_repair.risk_forced_delta.sum(dim=1)
            if sleeve_repair is not None
            else _risk_project(
                availability_ledger.weights,
                sequence.risk_asset_caps[fill],
                sequence.risk_gross_max[fill],
                sequence.cash_index,
            )
        )
        risk_ledger, accounting[TurnoverCause.RISK_FORCED] = availability_ledger.trade_to(
            risk_target, cause=TurnoverCause.RISK_FORCED
        )
        repaired_weights = risk_ledger.weights
        if sleeve_state is not None:
            self._assert_close(
                "H3 repaired aggregate", sleeve_state.aggregate_weights(), repaired_weights
            )

        trade_mask = (
            pending.decision_available
            & sequence.fill_membership[fill]
            & sequence.fill_availability[fill]
        )
        if sleeve_state is not None:
            assert pending.intent.entry_scores is not None
            sleeve_review = sleeve_state.review_maturing_(
                pending.intent.entry_scores,
                sequence.benchmark_weights[fill],
                trade_mask,
                sequence.risk_asset_caps[fill],
                sequence.risk_gross_max[fill],
            )
            built = _built_from_sleeve_review(
                sleeve_state,
                repaired_weights,
                sleeve_review,
            )
        else:
            built = self._build_action(
                pending.intent,
                risk_ledger,
                sequence.benchmark_weights[fill],
                trade_mask,
                sequence.risk_asset_caps[fill],
                sequence.risk_gross_max[fill],
            )
        safety_target = _risk_project(
            _force_mask_to_cash(
                _force_mask_to_cash(
                    built.target_weights,
                    sequence.fill_membership[fill],
                    sequence.cash_index,
                ),
                sequence.fill_availability[fill],
                sequence.cash_index,
            ),
            sequence.risk_asset_caps[fill],
            sequence.risk_gross_max[fill],
            sequence.cash_index,
        )
        filled_delta = safety_target - repaired_weights
        projection_distance = _one_way(filled_delta, built.constructed_delta)
        if not bool(torch.isfinite(projection_distance).all()):
            raise Hold30SafetyProjectionError("non-finite safety-projection distance")
        if bool((projection_distance > self.reconciliation_tolerance).any()):
            maximum = float(projection_distance.detach().max())
            raise Hold30SafetyProjectionError(
                "action construction was not fill-time feasible; "
                f"constructed-to-filled one-way distance {maximum:g} exceeds "
                f"{self.reconciliation_tolerance:g}"
            )
        if sleeve_state is not None:
            # The package-owned sleeve builder is designed to be feasible by
            # construction.  Preserve the actual safety-filled aggregate in
            # the maturing sleeve for sub-tolerance floating corrections.
            correction = safety_target - sleeve_state.aggregate_weights()
            if bool((correction != 0).any()):
                assert sleeve_review is not None
                books = sleeve_state.books.clone()
                books[:, sleeve_review.maturing_sleeve] = (
                    books[:, sleeve_review.maturing_sleeve] + correction
                )
                sleeve_state = Hold30SleeveState(
                    books,
                    session_index=sleeve_state.session_index,
                    last_review_session=sleeve_state.last_review_session,
                    review_count=sleeve_state.review_count,
                    cash_index=sleeve_state.cash_index,
                )

        proposed_release = (
            built.proposed_release_by_age if self.mechanism == "H2" else None
        )
        pre_cost_ledger, discretionary = risk_ledger.trade_to(
            safety_target,
            cause=TurnoverCause.DISCRETIONARY,
            proposed_release=proposed_release,
            track_new_entries=bool(sequence.entry_tracking_mask[t].item()),
        )
        accounting[TurnoverCause.DISCRETIONARY] = discretionary
        accounting[TurnoverCause.STARTUP] = _zero_accounting(pre_cost_ledger, TurnoverCause.STARTUP)
        accounting[TurnoverCause.TERMINAL] = _zero_accounting(pre_cost_ledger, TurnoverCause.TERMINAL)
        if set(accounting) != set(TURNOVER_CAUSES):
            raise AssertionError("transition did not classify every turnover cause")
        pre_cost_ledger.assert_reconciles(safety_target, atol=self.reconciliation_tolerance)

        turnover_by_cause = {cause: accounting[cause].turnover for cause in TURNOVER_CAUSES}
        total_turnover = torch.stack(tuple(turnover_by_cause.values()), dim=0).sum(0)
        cost = sequence.cost_rates[t] * total_turnover
        net_return = holding_return - cost
        if bool((net_return <= -1.0).any()):
            raise RuntimeError("net return reached -100%; check returns and cost assumptions")
        benchmark = sequence.benchmark_net_returns[t]
        utility = benchmark_relative_log_utility(net_return, benchmark)
        equity_after = state.equity * (1.0 + net_return)

        # The v1 linear cost is financed pro rata from the complete pre-cost
        # book.  In beginning-of-row return units, each coordinate finances
        # ``w_i * cost``; division by the resulting NAV leaves normalized
        # weights unchanged.  Cost financing changes NAV, not shares relative
        # to one another, and therefore is not a trade or a cohort exit.
        cost_financing = pre_cost_ledger.weights * cost.unsqueeze(-1)
        surviving_nav = 1.0 + holding_return - cost
        if bool((surviving_nav <= 0).any()):
            raise RuntimeError("cost financing exhausted portfolio NAV")
        # Retaining a separately cloned ledger makes the post-cost
        # normalization stage explicit and prevents it from being mistaken for
        # turnover/projection in persisted traces.
        post_cost_ledger = pre_cost_ledger.clone()
        post_cost_weights = post_cost_ledger.weights
        post_cost_ledger.assert_reconciles(post_cost_weights, atol=self.reconciliation_tolerance)

        next_state = Hold30RuntimeState(
            position_index=fill,
            ledger=post_cost_ledger,
            equity=equity_after,
            pending_intent=None,
            sleeve_snapshot=(
                None if sleeve_state is None else sleeve_state.capture()
            ),
        )
        transition = Hold30Transition(
            decision_index=t,
            fill_index=fill,
            raw_intent=pending.intent,
            decision_weights=decision_weights,
            execution_pretrade_weights=execution_pretrade_weights,
            retention_units_before_membership=execution_pretrade.retention_units,
            membership_repaired_weights=membership_ledger.weights,
            availability_repaired_weights=availability_ledger.weights,
            risk_repaired_weights=repaired_weights,
            retention_units_after_forced=risk_ledger.retention_units,
            requested_delta=built.requested_delta,
            constructed_delta=built.constructed_delta,
            filled_delta=filled_delta,
            pre_cost_weights=pre_cost_ledger.weights,
            post_cost_weights=post_cost_weights,
            holding_return=holding_return,
            cost=cost,
            cost_financing=cost_financing,
            net_return=net_return,
            benchmark_net_return=benchmark,
            utility=utility,
            projection_distance=projection_distance,
            turnover_by_cause=turnover_by_cause,
            accounting_by_cause=accounting,
            discretionary_accounting=discretionary,
            sleeve_repair=sleeve_repair,
            sleeve_review=sleeve_review,
            equity_before=state.equity,
            equity_after=equity_after,
        )
        return next_state, transition

    def canonical_pass(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
        roles: Hold30CreditRoles,
    ) -> tuple[Hold30CanonicalTrace, Sequence[Hold30CanonicalRow]]:
        """Run one continuous no-grad deployed chronology through terminal."""

        if torch.is_grad_enabled():
            raise RuntimeError("canonical_pass must run under torch.no_grad()")
        if roles.n_positions != sequence.n_positions:
            raise ValueError("credit roles do not match the sequence length")
        decision_states = self._canonical_decision_states(policy, sequence)
        state = self.initial_state(sequence)
        boundaries: list[Hold30RuntimeState] = [state.detach()]
        pendings: list[PendingHold30Intent] = []
        transitions: list[Hold30Transition] = []
        rows: list[Hold30CanonicalRow] = []
        while state.position_index < sequence.n_positions - 1:
            state = self.decide(
                policy,
                sequence,
                state,
                decision_state=decision_states[state.position_index],
            )
            assert state.pending_intent is not None
            pendings.append(state.pending_intent.detach())
            state, transition = self.advance(sequence, state)
            transitions.append(_detach_transition(transition))
            boundaries.append(state.detach())
            rows.append(self._canonical_row(transition))
        if state.pending_intent is not None:
            raise AssertionError("terminal state retained a pending intent")
        trace = Hold30CanonicalTrace(
            tuple(boundaries),
            tuple(value.detach().clone() for value in decision_states),
            tuple(pendings),
            tuple(transitions),
        )
        return trace, tuple(rows)

    def replay_origins(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
        canonical_state: Hold30CanonicalTrace,
        origins: torch.Tensor,
        roles: Hold30CreditRoles,
    ) -> Sequence[Hold30OriginReplay]:
        """Restore and replay independent anchors with one attached intent each."""

        if roles.n_positions != sequence.n_positions:
            raise ValueError("credit roles do not match the sequence length")
        origins = torch.as_tensor(origins, dtype=torch.long, device="cpu")
        if origins.ndim != 1 or origins.numel() == 0:
            raise ValueError("origins must be a non-empty one-dimensional tensor")
        anchor_set = {int(value) for value in roles.anchors}
        origin_values = tuple(int(value) for value in origins.tolist())
        for origin in origin_values:
            if origin not in anchor_set:
                raise ValueError(f"origin {origin} is not a loss-bearing anchor")
        origin_states = self._replay_decision_states(policy, sequence, origins)
        if len(origin_states) != len(origin_values):
            raise AssertionError("state provider changed the origin batch length")

        result: list[Hold30OriginReplay] = []
        for origin, origin_state in zip(origin_values, origin_states, strict=True):
            terminal = origin + 31
            state = canonical_state.boundary_states[origin].detach()
            self._validate_decision_state(origin_state, sequence)
            self._assert_close(
                "decision state",
                origin_state,
                canonical_state.decision_states[origin],
            )
            state = self.decide(
                policy,
                sequence,
                state,
                decision_state=origin_state,
            )
            origin_gate, origin_entropy = self._gate_metrics(state.pending_intent)
            replayed: list[Hold30Transition] = []
            while state.position_index < terminal:
                row = state.position_index
                if row != origin:
                    state = self.attach_pending(
                        sequence,
                        state,
                        canonical_state.pending_intents[row].detach(),
                    )
                state, transition = self.advance(sequence, state)
                self._assert_transition_equal(transition, canonical_state.transitions[row])
                self._assert_state_equal(state, canonical_state.boundary_states[row + 1])
                replayed.append(transition)
            if len(replayed) != 31:
                raise AssertionError("origin replay must contain one fill row and thirty return rows")
            origin_transition = replayed[0]
            result.append(
                Hold30OriginReplay(
                    origin=origin,
                    utility_rows=torch.stack([item.utility.mean() for item in replayed]),
                    discretionary_turnover=origin_transition.discretionary_accounting.turnover.mean(),
                    early_sale_mass=origin_transition.discretionary_accounting.early_exit_notional.mean(),
                    gate=origin_gate,
                    gate_entropy=origin_entropy,
                )
            )
        return tuple(result)

    def run_to_terminal(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
        state: Hold30RuntimeState | None = None,
    ) -> tuple[Hold30RuntimeState, tuple[Hold30Transition, ...]]:
        """Continue a fresh or receipt-restored state without economic reset."""

        current = self.initial_state(sequence) if state is None else state
        decision_states = self._canonical_decision_states(policy, sequence)
        transitions: list[Hold30Transition] = []
        while current.position_index < sequence.n_positions - 1:
            if current.pending_intent is None:
                current = self.decide(
                    policy,
                    sequence,
                    current,
                    decision_state=decision_states[current.position_index],
                )
            current, transition = self.advance(sequence, current)
            transitions.append(transition)
        return current, tuple(transitions)

    def _build_action(
        self,
        intent: Hold30Intent,
        repaired_ledger: CohortLedger,
        benchmark_weights: torch.Tensor,
        trade_mask: torch.Tensor,
        risk_asset_caps: torch.Tensor,
        risk_gross_max: torch.Tensor,
    ) -> Hold30BuiltAction:
        if self.action_builder is not None:
            built = self.action_builder(
                intent,
                repaired_ledger,
                benchmark_weights,
                trade_mask,
                risk_asset_caps,
                risk_gross_max,
            )
        elif self.mechanism in {"H0", "H1"}:
            assert intent.target_logits is not None and intent.gate is not None
            built = build_scalar_gate_hold30_action(
                repaired_ledger.weights,
                intent.target_logits,
                intent.gate,
                benchmark_weights,
                trade_mask,
                risk_asset_caps,
                risk_gross_max,
                cash_index=repaired_ledger.cash_index,
            )
        elif self.mechanism == "H2":
            assert intent.entry_scores is not None
            assert intent.hazard_residual is not None
            assert intent.exposure_residual is not None
            if intent.active_risk_scale is not None:
                built = build_alpha_hold30_action(
                    repaired_ledger.weights,
                    repaired_ledger.economic_value,
                    intent.entry_scores,
                    intent.hazard_residual,
                    intent.active_risk_scale,
                    benchmark_weights,
                    trade_mask,
                    risk_asset_caps,
                    risk_gross_max,
                    # V5 exposes the hard straight-through branch explicitly;
                    # legacy generations keep the original field name.
                    exact_hold_probability=(
                        intent.exact_hold_decision_st
                        if intent.exact_hold_decision_st is not None
                        else intent.exact_hold_probability
                    ),
                    total_risk_overlay=intent.total_risk_overlay,
                    total_risk_step=self.alpha_total_risk_step,
                    cash_index=repaired_ledger.cash_index,
                )
            else:
                built = build_h2_hold30_action(
                    repaired_ledger.weights,
                    repaired_ledger.economic_value,
                    intent.entry_scores,
                    intent.hazard_residual,
                    intent.exposure_residual,
                    benchmark_weights,
                    trade_mask,
                    risk_asset_caps,
                    risk_gross_max,
                    exact_hold_probability=intent.exact_hold_probability,
                    cash_index=repaired_ledger.cash_index,
                )
        else:  # guarded by __init__
            raise AssertionError("H3 action builder is missing")
        if not isinstance(built, Hold30BuiltAction):
            raise TypeError("action_builder must return Hold30BuiltAction")
        return built

    def _validate_intent(self, intent: Hold30Intent, batch: int, assets: int) -> None:
        matrix_shape = (batch, assets)
        vector_shape = (batch,)
        prohibited: tuple[str, ...]
        if self.mechanism in {"H0", "H1"}:
            required = {"target_logits": matrix_shape, "gate": vector_shape}
            prohibited = (
                "entry_scores",
                "hazard_residual",
                "raw_hazard_residual",
                "exact_hold_probability",
                "exact_hold_logit",
                "exact_hold_soft_probability",
                "exact_hold_decision_st",
                "exposure_residual",
            )
        elif self.mechanism == "H2":
            required = {
                "entry_scores": matrix_shape,
                "hazard_residual": matrix_shape,
                "exposure_residual": vector_shape,
            }
            prohibited = ("target_logits", "gate")
        else:
            required = {"entry_scores": matrix_shape}
            prohibited = (
                "target_logits",
                "gate",
                "hazard_residual",
                "raw_hazard_residual",
                "exact_hold_probability",
                "exact_hold_logit",
                "exact_hold_soft_probability",
                "exact_hold_decision_st",
                "exposure_residual",
            )
        for name, shape in required.items():
            value = getattr(intent, name)
            if value is None or tuple(value.shape) != shape:
                raise ValueError(f"{self.mechanism} intent field {name} must have shape {shape}")
        for name in prohibited:
            if getattr(intent, name) is not None:
                raise ValueError(f"{self.mechanism} intent must not populate {name}")
        if self.mechanism == "H2":
            for name in (
                "raw_hazard_residual",
                "exact_hold_probability",
                "exact_hold_logit",
                "exact_hold_soft_probability",
                "exact_hold_decision_st",
            ):
                value = getattr(intent, name)
                if value is not None and tuple(value.shape) != matrix_shape:
                    raise ValueError(
                        f"H2 intent field {name} must have shape {matrix_shape} when present"
                    )
            if intent.exact_hold_probability is not None and bool(
                (
                    (intent.exact_hold_probability < 0)
                    | (intent.exact_hold_probability > 1)
                ).any()
            ):
                raise ValueError("exact_hold_probability must lie in [0,1]")
            if intent.exact_hold_soft_probability is not None and bool(
                (intent.exact_hold_soft_probability < 0).any()
                or (intent.exact_hold_soft_probability > 1).any()
            ):
                raise ValueError("exact_hold_soft_probability must lie in [0,1]")
            if intent.exact_hold_decision_st is not None and bool(
                ((intent.exact_hold_decision_st != 0) & (intent.exact_hold_decision_st != 1)).any()
            ):
                raise ValueError("exact_hold_decision_st must be hard binary")
            if intent.exact_hold_probability is not None and any(
                value is not None
                for value in (
                    intent.exact_hold_logit,
                    intent.exact_hold_soft_probability,
                    intent.exact_hold_decision_st,
                )
            ):
                raise ValueError(
                    "legacy exact_hold_probability and explicit v5 exact-hold fields are mutually exclusive"
                )
        alpha_fields = (
            "alpha_mean_30d",
            "alpha_downside_30d",
            "active_risk_scale",
            "signal_confidence",
            "uncalibrated_signal_confidence_logit",
            "benchmark_derisk_request",
            "total_risk_overlay",
            "auxiliary_alpha_mean",
        )
        if intent.active_risk_scale is None:
            if any(getattr(intent, name) is not None for name in alpha_fields):
                raise ValueError("partial v3 alpha intent is forbidden")
            return
        if self.mechanism != "H2":
            raise ValueError("v3 alpha intent requires the age-aware H2 runtime")
        if intent.alpha_mean_30d is None or tuple(
            intent.alpha_mean_30d.shape
        ) != matrix_shape:
            raise ValueError(
                f"v3 alpha intent field alpha_mean_30d must have shape {matrix_shape}"
            )
        if intent.alpha_downside_30d is not None and tuple(
            intent.alpha_downside_30d.shape
        ) != matrix_shape:
            raise ValueError(
                f"v3 alpha_downside_30d must have shape {matrix_shape} when present"
            )
        if tuple(intent.active_risk_scale.shape) != vector_shape:
            raise ValueError("v3 active_risk_scale must have shape [batch]")
        if intent.signal_confidence is not None:
            if tuple(intent.signal_confidence.shape) != vector_shape:
                raise ValueError("M03R signal_confidence must have shape [batch]")
            if bool(
                ((intent.signal_confidence < 0) | (intent.signal_confidence > 1)).any()
            ):
                raise ValueError("M03R signal_confidence must lie in [0,1]")
        if intent.uncalibrated_signal_confidence_logit is not None and tuple(
            intent.uncalibrated_signal_confidence_logit.shape
        ) != vector_shape:
            raise ValueError(
                "M03R uncalibrated_signal_confidence_logit must have shape [batch]"
            )
        if intent.benchmark_derisk_request is not None:
            if tuple(intent.benchmark_derisk_request.shape) != vector_shape:
                raise ValueError(
                    "M03R benchmark_derisk_request must have shape [batch]"
                )
            if bool(
                (intent.benchmark_derisk_request < 0).any()
                or (intent.benchmark_derisk_request > 1).any()
            ):
                raise ValueError("M03R benchmark_derisk_request must lie in [0,1]")
        if intent.total_risk_overlay is not None and tuple(
            intent.total_risk_overlay.shape
        ) != vector_shape:
            raise ValueError("v3 total_risk_overlay must have shape [batch]")
        if intent.auxiliary_alpha_mean is None or tuple(
            intent.auxiliary_alpha_mean.shape
        ) != (*matrix_shape, len(HOLD30_ALPHA_HORIZONS)):
            raise ValueError(
                "v3 auxiliary_alpha_mean must have shape "
                f"[batch,asset,{len(HOLD30_ALPHA_HORIZONS)}]"
            )

    def _validate_state(self, sequence: Hold30Sequence, state: Hold30RuntimeState) -> None:
        if not isinstance(state.position_index, int) or not 0 <= state.position_index < sequence.n_positions:
            raise ValueError("runtime position_index is outside the sequence")
        if state.ledger.batch_size != sequence.batch_size or state.ledger.num_assets != sequence.num_assets:
            raise ValueError("runtime ledger does not match sequence axes")
        if state.ledger.cash_index != sequence.cash_index:
            raise ValueError("runtime CASH coordinate does not match sequence")
        _check_floating("runtime equity", state.equity, (sequence.batch_size,))
        state.ledger.assert_reconciles(state.ledger.weights, atol=self.reconciliation_tolerance)
        if self.mechanism == "H3":
            if state.sleeve_snapshot is None:
                raise ValueError("H3 runtime state is missing its sleeve snapshot")
            sleeves = Hold30SleeveState.from_snapshot(state.sleeve_snapshot)
            if sleeves.session_index != state.position_index:
                raise ValueError("H3 sleeve phase does not match runtime position")
            self._assert_close(
                "H3 state aggregate", sleeves.aggregate_weights(), state.ledger.weights
            )
        elif state.sleeve_snapshot is not None:
            raise ValueError("only H3 runtime state may contain a sleeve snapshot")

    def _canonical_decision_states(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
    ) -> tuple[torch.Tensor, ...]:
        values = self.state_provider.canonical_states(policy, sequence)
        if isinstance(values, torch.Tensor):
            if values.ndim < 4 or values.shape[0] != sequence.n_positions - 1:
                raise ValueError(
                    "state_provider canonical tensor must have shape "
                    "[n_positions-1, batch, asset, ...]"
                )
            states = tuple(values[index] for index in range(values.shape[0]))
        else:
            states = tuple(values)
            if len(states) != sequence.n_positions - 1:
                raise ValueError("state_provider must return one state per decision position")
        for value in states:
            self._validate_decision_state(value, sequence)
        return states

    def _replay_decision_states(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
        origins: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        values = self.state_provider.replay_origin_states(policy, sequence, origins)
        expected = int(origins.numel())
        if isinstance(values, torch.Tensor):
            if values.ndim < 4 or values.shape[0] != expected:
                raise ValueError(
                    "state_provider replay tensor must have shape "
                    "[origin, batch, asset, ...]"
                )
            states = tuple(values[index] for index in range(expected))
        else:
            states = tuple(values)
            if len(states) != expected:
                raise ValueError("state_provider must return one state per replay origin")
        for value in states:
            self._validate_decision_state(value, sequence)
        return states

    @staticmethod
    def _validate_decision_state(value: torch.Tensor, sequence: Hold30Sequence) -> None:
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim < 3
            or tuple(value.shape[:2]) != (sequence.batch_size, sequence.num_assets)
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(
                "decision state must be a finite floating tensor with leading "
                f"shape {(sequence.batch_size, sequence.num_assets)}"
            )

    def _canonical_row(self, transition: Hold30Transition) -> Hold30CanonicalRow:
        gate, entropy = self._gate_metrics_from_intent(transition.raw_intent)
        return Hold30CanonicalRow(
            utility=float(transition.utility.detach().mean()),
            discretionary_turnover=float(
                transition.discretionary_accounting.turnover.detach().mean()
            ),
            early_sale_mass=float(
                transition.discretionary_accounting.early_exit_notional.detach().mean()
            ),
            gate=float(gate.detach()),
            gate_entropy=float(entropy.detach()),
        )

    @staticmethod
    def _gate_metrics(pending: PendingHold30Intent | None) -> tuple[torch.Tensor, torch.Tensor]:
        if pending is None:
            raise AssertionError("origin decision did not create a pending intent")
        return Hold30ChronologicalRuntime._gate_metrics_from_intent(pending.intent)

    @staticmethod
    def _gate_metrics_from_intent(intent: Hold30Intent) -> tuple[torch.Tensor, torch.Tensor]:
        if intent.gate is None:
            reference = next((value for _, value in _intent_tensors(intent)), None)
            if reference is None:
                raise ValueError("empty Hold30Intent")
            zero = reference.new_zeros(())
            return zero, zero
        gate = intent.gate.mean()
        eps = torch.finfo(intent.gate.dtype).eps
        bounded = intent.gate.clamp(min=eps, max=1.0 - eps)
        entropy = (-(bounded * bounded.log() + (1.0 - bounded) * (1.0 - bounded).log())).mean()
        return gate, entropy

    def _assert_state_equal(self, actual: Hold30RuntimeState, expected: Hold30RuntimeState) -> None:
        if actual.position_index != expected.position_index or actual.pending_intent is not None:
            raise RuntimeError("canonical/replay boundary-state mismatch")
        self._assert_close("equity", actual.equity, expected.equity)
        self._assert_close(
            "cohort economic value", actual.ledger.economic_value, expected.ledger.economic_value
        )
        self._assert_close(
            "cohort retention units", actual.ledger.retention_units, expected.ledger.retention_units
        )
        if (actual.sleeve_snapshot is None) != (expected.sleeve_snapshot is None):
            raise RuntimeError("canonical/replay H3 sleeve-state presence mismatch")
        if actual.sleeve_snapshot is not None and expected.sleeve_snapshot is not None:
            if (
                actual.sleeve_snapshot.session_index != expected.sleeve_snapshot.session_index
                or actual.sleeve_snapshot.cash_index != expected.sleeve_snapshot.cash_index
            ):
                raise RuntimeError("canonical/replay H3 sleeve phase mismatch")
            self._assert_close(
                "H3 sleeve books",
                actual.sleeve_snapshot.books,
                expected.sleeve_snapshot.books,
            )
            if not torch.equal(
                actual.sleeve_snapshot.last_review_session,
                expected.sleeve_snapshot.last_review_session,
            ) or not torch.equal(
                actual.sleeve_snapshot.review_count,
                expected.sleeve_snapshot.review_count,
            ):
                raise RuntimeError("canonical/replay H3 sleeve counters mismatch")

    def _assert_transition_equal(self, actual: Hold30Transition, expected: Hold30Transition) -> None:
        if actual.decision_index != expected.decision_index or actual.fill_index != expected.fill_index:
            raise RuntimeError("canonical/replay transition index mismatch")
        for name in (
            "decision_weights",
            "execution_pretrade_weights",
            "retention_units_before_membership",
            "membership_repaired_weights",
            "availability_repaired_weights",
            "risk_repaired_weights",
            "retention_units_after_forced",
            "requested_delta",
            "constructed_delta",
            "filled_delta",
            "pre_cost_weights",
            "post_cost_weights",
            "holding_return",
            "cost",
            "cost_financing",
            "net_return",
            "benchmark_net_return",
            "utility",
            "projection_distance",
        ):
            self._assert_close(name, getattr(actual, name), getattr(expected, name))
        actual_intent = dict(_intent_tensors(actual.raw_intent))
        expected_intent = dict(_intent_tensors(expected.raw_intent))
        if actual_intent.keys() != expected_intent.keys():
            raise RuntimeError("canonical/replay raw-intent field mismatch")
        for name, value in actual_intent.items():
            self._assert_close(f"intent.{name}", value, expected_intent[name])
        for cause in TURNOVER_CAUSES:
            self._assert_close(
                f"turnover.{cause.value}",
                actual.turnover_by_cause[cause],
                expected.turnover_by_cause[cause],
            )
            actual_cause = actual.accounting_by_cause[cause]
            expected_cause = expected.accounting_by_cause[cause]
            for field in (
                "turnover",
                "net_buys",
                "net_sells",
                "sold_value_by_age",
                "sold_units_by_age",
                "entry_units_added",
                "early_exit_notional",
                "early_exit_units",
            ):
                self._assert_close(
                    f"accounting.{cause.value}.{field}",
                    getattr(actual_cause, field),
                    getattr(expected_cause, field),
                )
        if (actual.sleeve_review is None) != (expected.sleeve_review is None):
            raise RuntimeError("canonical/replay H3 review presence mismatch")
        if (actual.sleeve_repair is None) != (expected.sleeve_repair is None):
            raise RuntimeError("canonical/replay H3 repair presence mismatch")
        if actual.sleeve_repair is not None and expected.sleeve_repair is not None:
            for field in (
                "membership_forced_delta",
                "availability_forced_delta",
                "risk_forced_delta",
            ):
                self._assert_close(
                    f"H3 repair.{field}",
                    getattr(actual.sleeve_repair, field),
                    getattr(expected.sleeve_repair, field),
                )
        if actual.sleeve_review is not None and expected.sleeve_review is not None:
            if (
                actual.sleeve_review.session_index != expected.sleeve_review.session_index
                or actual.sleeve_review.maturing_sleeve != expected.sleeve_review.maturing_sleeve
                or actual.sleeve_review.review_age != expected.sleeve_review.review_age
            ):
                raise RuntimeError("canonical/replay H3 review phase mismatch")
            for field in _SLEEVE_REVIEW_TENSOR_FIELDS:
                self._assert_close(
                    f"H3 review.{field}",
                    getattr(actual.sleeve_review, field),
                    getattr(expected.sleeve_review, field),
                )

    def _assert_close(self, name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
        if not actual.is_floating_point() or not expected.is_floating_point():
            if not torch.equal(actual.detach(), expected.detach()):
                raise RuntimeError(f"canonical/replay {name} mismatch")
            return
        if not bool(
            torch.allclose(
                actual.detach(),
                expected.detach(),
                atol=self.reconciliation_tolerance,
                rtol=self.reconciliation_tolerance,
            )
        ):
            maximum = float((actual.detach() - expected.detach()).abs().max())
            raise RuntimeError(
                f"canonical/replay {name} mismatch (maximum absolute difference {maximum:g})"
            )


class Hold30ChronologicalReplayAdapter:
    """Thin protocol adapter around :class:`Hold30ChronologicalRuntime`."""

    def __init__(self, runtime: Hold30ChronologicalRuntime) -> None:
        self.runtime = runtime

    def canonical_pass(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
        roles: Hold30CreditRoles,
    ) -> tuple[Hold30CanonicalTrace, Sequence[Hold30CanonicalRow]]:
        return self.runtime.canonical_pass(policy, sequence, roles)

    def replay_origins(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
        canonical_state: Hold30CanonicalTrace,
        origins: torch.Tensor,
        roles: Hold30CreditRoles,
    ) -> Sequence[Hold30OriginReplay]:
        return self.runtime.replay_origins(policy, sequence, canonical_state, origins, roles)


def _detach_accounting(value: CohortTradeAccounting) -> CohortTradeAccounting:
    return CohortTradeAccounting(
        cause=value.cause,
        turnover=value.turnover.detach().clone(),
        net_buys=value.net_buys.detach().clone(),
        net_sells=value.net_sells.detach().clone(),
        sold_value_by_age=value.sold_value_by_age.detach().clone(),
        sold_units_by_age=value.sold_units_by_age.detach().clone(),
        entry_units_added=value.entry_units_added.detach().clone(),
        early_exit_notional=value.early_exit_notional.detach().clone(),
        early_exit_units=value.early_exit_units.detach().clone(),
    )


_SLEEVE_REVIEW_TENSOR_FIELDS = (
    "sleeve_nav_before",
    "entry_direction",
    "residual_asset_capacity",
    "residual_risky_capacity",
    "target_sleeve",
    "requested_sleeve_delta",
    "constructed_sleeve_delta",
    "requested_delta",
    "constructed_delta",
    "requested_turnover",
    "constructed_turnover",
    "same_name_cross_net_notional",
    "unallocated_sleeve_cash",
    "maturity_cap_censored",
)


def _detach_sleeve_repair(value: Hold30SleeveRepair | None) -> Hold30SleeveRepair | None:
    if value is None:
        return None
    return Hold30SleeveRepair(
        membership_forced_delta=value.membership_forced_delta.detach().clone(),
        availability_forced_delta=value.availability_forced_delta.detach().clone(),
        risk_forced_delta=value.risk_forced_delta.detach().clone(),
    )


def _detach_sleeve_review(value: Hold30SleeveReview | None) -> Hold30SleeveReview | None:
    if value is None:
        return None
    fields = {
        name: getattr(value, name).detach().clone()
        for name in _SLEEVE_REVIEW_TENSOR_FIELDS
    }
    return Hold30SleeveReview(
        session_index=value.session_index,
        maturing_sleeve=value.maturing_sleeve,
        review_age=value.review_age,
        **fields,
    )


def _detach_transition(value: Hold30Transition) -> Hold30Transition:
    tensors = {
        name: getattr(value, name).detach().clone()
        for name in (
            "decision_weights",
            "execution_pretrade_weights",
            "retention_units_before_membership",
            "membership_repaired_weights",
            "availability_repaired_weights",
            "risk_repaired_weights",
            "retention_units_after_forced",
            "requested_delta",
            "constructed_delta",
            "filled_delta",
            "pre_cost_weights",
            "post_cost_weights",
            "holding_return",
            "cost",
            "cost_financing",
            "net_return",
            "benchmark_net_return",
            "utility",
            "projection_distance",
            "equity_before",
            "equity_after",
        )
    }
    return Hold30Transition(
        decision_index=value.decision_index,
        fill_index=value.fill_index,
        raw_intent=_clone_intent(value.raw_intent, detach=True),
        turnover_by_cause={
            cause: amount.detach().clone() for cause, amount in value.turnover_by_cause.items()
        },
        accounting_by_cause={
            cause: _detach_accounting(accounting)
            for cause, accounting in value.accounting_by_cause.items()
        },
        discretionary_accounting=_detach_accounting(value.discretionary_accounting),
        sleeve_repair=_detach_sleeve_repair(value.sleeve_repair),
        sleeve_review=_detach_sleeve_review(value.sleeve_review),
        **tensors,
    )


__all__ = [
    "HOLD30_RECONCILIATION_TOLERANCE",
    "FunctionalHold30DecisionStateProvider",
    "Hold30ActionBuilder",
    "Hold30CanonicalTrace",
    "Hold30ChronologicalReplayAdapter",
    "Hold30ChronologicalRuntime",
    "Hold30DecisionStateProvider",
    "Hold30Policy",
    "Hold30RuntimeState",
    "Hold30SafetyProjectionError",
    "Hold30Sequence",
    "Hold30Transition",
    "PendingHold30Intent",
    "TensorHold30DecisionStateProvider",
]
