"""Leakage-safe economic view of the 2026 TOP2000 chronology.

The policy encoder may consume the complete causal 252-state pre-2026 history,
including observations that belonged to a checkpoint's training fold.  The
portfolio ledger is a separate boundary: it starts from C1 only on the first
selected decision whose *next* return is outside that checkpoint's training
state interval.  Consequently an in-sample policy decision can never create a
holding that is carried into the 2026 score window.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, cast

import torch

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.evaluation.top2000_m03r_v7_2026_retrospective_data import (
    TOP2000_M03R_V7_2026_CONTEXT_STATES,
    Top2000M03RV72026RetrospectiveData,
)
from rl_quant.training.hold30_runtime import (
    Hold30DecisionStateProvider,
    Hold30Policy,
    Hold30Sequence,
)
from rl_quant.training.hold30_top2000_development import (
    Top2000VerifiedDevelopmentCache,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS,
    Top2000M03RV7DecisionStateProvider,
    Top2000M03RV7DevelopmentFold,
    Top2000M03RV7DevelopmentPolicy,
    bind_top2000_m03r_v7_runtime_sequence,
)

TOP2000_M03R_V7_2026_EXECUTION_VIEW_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-economic-execution-view-v1"
)


class Top2000M03RV72026ExecutionViewError(ValueError):
    """The encoder/economic boundary is absent, inconsistent, or leaky."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Top2000M03RV72026ExecutionViewError(
            "execution-view receipt is not canonical-JSON safe"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(_canonical_json(list(tensor.shape)))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _require_digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Top2000M03RV72026ExecutionViewError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026EconomicExecutionReceipt:
    """Exact split between causal encoder history and portfolio economics."""

    chronology_receipt_sha256: str
    pre2026_cache_sha256: str
    pre2026_cache_identity: str
    training_fold_receipt_sha256: str
    training_fold_index: int
    training_state_stop_exclusive: int
    training_cutoff_state_index: int
    training_cutoff_date: str
    selected_context_cache_start_index: int
    selected_context_start_date: str
    economic_execution_start: int
    economic_execution_cache_state_index: int
    economic_execution_decision_date: str
    first_economic_return_date: str
    global_score_transition_start: int
    global_score_transition_stop_exclusive: int
    local_score_transition_start: int
    local_score_transition_stop_exclusive: int
    encoder_input_state_rows: int
    encoder_only_economic_prefix_transition_rows: int
    executed_state_rows: int
    executed_transition_rows: int
    initial_c1_weights_sha256: str
    initial_ledger_economic_value_sha256: str
    initial_ledger_retention_units_sha256: str
    execution_axis_id: str
    initial_ledger_age: int = 0
    initial_ledger_source: str = "C1-at-economic-execution-start"
    next_return_strictly_outside_training_states: bool = True
    learned_policy_actions_before_execution_start: int = 0
    in_sample_origin_holdings_enter_2026: bool = False
    full_encoder_history_is_economic_history: bool = False
    one_continuous_economic_trace_into_2026: bool = True
    state_reset_count_within_economic_trace: int = 0
    policy_training_authorized: bool = False
    development_only: bool = True
    future_selected_universe: bool = True
    scientific_reporting_eligible: bool = False
    promotion_eligible: bool = False
    schema: str = TOP2000_M03R_V7_2026_EXECUTION_VIEW_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "chronology_receipt_sha256",
            "pre2026_cache_sha256",
            "pre2026_cache_identity",
            "training_fold_receipt_sha256",
            "initial_c1_weights_sha256",
            "initial_ledger_economic_value_sha256",
            "initial_ledger_retention_units_sha256",
        ):
            _require_digest(name, getattr(self, name))
        if (
            self.schema != TOP2000_M03R_V7_2026_EXECUTION_VIEW_SCHEMA
            or self.training_fold_index not in range(6)
            or self.training_state_stop_exclusive
            != self.training_cutoff_state_index + 1
            or self.selected_context_cache_start_index < 0
            or self.economic_execution_cache_state_index
            != self.selected_context_cache_start_index
            + self.economic_execution_start
            or not 0
            <= self.economic_execution_start
            <= self.global_score_transition_start
            or self.local_score_transition_start
            != self.global_score_transition_start - self.economic_execution_start
            or self.local_score_transition_stop_exclusive
            != self.global_score_transition_stop_exclusive
            - self.economic_execution_start
            or self.encoder_only_economic_prefix_transition_rows
            != self.economic_execution_start
            or self.executed_state_rows != self.executed_transition_rows + 1
            or self.executed_transition_rows
            != self.encoder_input_state_rows - 1 - self.economic_execution_start
            or self.initial_ledger_age != 0
            or self.initial_ledger_source != "C1-at-economic-execution-start"
            or not self.execution_axis_id
            or not self.next_return_strictly_outside_training_states
            or self.learned_policy_actions_before_execution_start != 0
            or self.in_sample_origin_holdings_enter_2026
            or self.full_encoder_history_is_economic_history
            or not self.one_continuous_economic_trace_into_2026
            or self.state_reset_count_within_economic_trace != 0
            or self.policy_training_authorized
            or not self.development_only
            or not self.future_selected_universe
            or self.scientific_reporting_eligible
            or self.promotion_eligible
        ):
            raise Top2000M03RV72026ExecutionViewError(
                "execution-view geometry or research-only semantics drifted"
            )
        if not (
            self.training_cutoff_date
            <= self.economic_execution_decision_date
            < self.first_economic_return_date
        ):
            raise Top2000M03RV72026ExecutionViewError(
                "economic execution must begin on/after cutoff with a later return date"
            )

    def canonical_payload(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_payload())


@dataclass(frozen=True, slots=True)
class _EncoderHistoryEconomicSuffixProvider:
    """Encode the full causal history, expose only post-cutoff decision rows."""

    base: Top2000M03RV7DecisionStateProvider
    full_sequence: Hold30Sequence
    economic_execution_start: int
    trains_upstream_encoder: bool = False

    def _full_states(self, policy: Hold30Policy) -> torch.Tensor:
        values = self.base.canonical_states(policy, self.full_sequence)
        if not isinstance(values, torch.Tensor):
            raise Top2000M03RV72026ExecutionViewError(
                "TOP2000 encoder provider must return one tensor chronology"
            )
        return values

    def canonical_states(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
    ) -> torch.Tensor:
        values = self._full_states(policy)[self.economic_execution_start :]
        if values.shape[0] != sequence.n_positions - 1:
            raise Top2000M03RV72026ExecutionViewError(
                "encoder suffix does not match the economic sequence"
            )
        return values

    def replay_origin_state(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
        origin: int,
    ) -> torch.Tensor:
        del sequence
        return self._full_states(policy)[self.economic_execution_start + origin]

    def replay_origin_states(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
        origins: torch.Tensor,
    ) -> torch.Tensor:
        del sequence
        indexes = origins.to(dtype=torch.long) + self.economic_execution_start
        values = self._full_states(policy)
        return values.index_select(0, indexes.to(device=values.device))


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026EconomicExecutionView:
    """Runtime sequence plus a full-history/suffix-only state provider."""

    sequence: Hold30Sequence
    state_provider: Hold30DecisionStateProvider
    receipt: Top2000M03RV72026EconomicExecutionReceipt

    def __post_init__(self) -> None:
        if (
            self.sequence.axis_id != self.receipt.execution_axis_id
            or self.sequence.n_positions != self.receipt.executed_state_rows
            or self.sequence.n_positions - 1 != self.receipt.executed_transition_rows
            or self.sequence.initial_ledger.cash_index != 0
        ):
            raise Top2000M03RV72026ExecutionViewError(
                "economic sequence does not match its execution receipt"
            )
        initial = self.sequence.initial_ledger
        if (
            _tensor_sha256(initial.weights)
            != self.receipt.initial_c1_weights_sha256
            or _tensor_sha256(initial.economic_value)
            != self.receipt.initial_ledger_economic_value_sha256
            or _tensor_sha256(initial.retention_units)
            != self.receipt.initial_ledger_retention_units_sha256
            or bool((initial.economic_value[..., 1:] != 0).any())
            or bool((initial.retention_units != 0).any())
        ):
            raise Top2000M03RV72026ExecutionViewError(
                "initial C1 ledger is not an untracked age-zero post-cutoff endowment"
            )


def build_top2000_m03r_v7_2026_economic_execution_view(
    retrospective: Top2000M03RV72026RetrospectiveData,
    pre2026_cache: Top2000VerifiedDevelopmentCache,
    training_fold: Top2000M03RV7DevelopmentFold,
    policy: Top2000M03RV7DevelopmentPolicy,
) -> Top2000M03RV72026EconomicExecutionView:
    """Bind full encoder history while starting economics at an OOS return."""

    if not isinstance(retrospective, Top2000M03RV72026RetrospectiveData):
        raise Top2000M03RV72026ExecutionViewError(
            "execution view requires typed retrospective data"
        )
    if not isinstance(pre2026_cache, Top2000VerifiedDevelopmentCache):
        raise Top2000M03RV72026ExecutionViewError(
            "execution view requires the verified pre-2026 cache"
        )
    if not isinstance(training_fold, Top2000M03RV7DevelopmentFold):
        raise Top2000M03RV72026ExecutionViewError(
            "execution view requires a typed training fold"
        )
    pre2026_cache.validate_unmodified()
    identity = retrospective.identity
    selected_cache_start = (
        TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS
        - TOP2000_M03R_V7_2026_CONTEXT_STATES
    )
    if (
        len(pre2026_cache.exchange_dates)
        != TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS
        or pre2026_cache.cache_sha256 != identity.pre2026_cache_sha256
        or pre2026_cache.cache_identity != identity.pre2026_cache_identity
        or pre2026_cache.action_hash != identity.action_hash
        or pre2026_cache.action_ids != retrospective.action_ids
        or pre2026_cache.exchange_dates[selected_cache_start:]
        != retrospective.exchange_dates[:TOP2000_M03R_V7_2026_CONTEXT_STATES]
    ):
        raise Top2000M03RV72026ExecutionViewError(
            "training-cache suffix does not reproduce the bound encoder context"
        )
    cutoff_index = training_fold.training_state_stop_exclusive - 1
    execution_cache_index = max(selected_cache_start, cutoff_index)
    execution_start = execution_cache_index - selected_cache_start
    if (
        execution_start > identity.score_transition_start
        or execution_cache_index + 1
        < training_fold.training_state_stop_exclusive
    ):
        raise Top2000M03RV72026ExecutionViewError(
            "economic start does not make its first return strictly out of sample"
        )

    full_bound, base_provider = bind_top2000_m03r_v7_runtime_sequence(
        retrospective.sequence,
        policy,
        placeholder_token_dim=1,
    )
    initial_weights = full_bound.benchmark_weights[execution_start]
    initial_ledger = CohortLedger.from_weights(
        initial_weights,
        cash_index=full_bound.cash_index,
        initial_age=0,
        track_initial_units=False,
    )
    axis_basis = _sha256(
        {
            "chronology_receipt_sha256": identity.receipt_sha256,
            "training_fold_receipt_sha256": training_fold.receipt_sha256,
            "economic_execution_start": execution_start,
            "initial_c1_weights_sha256": _tensor_sha256(initial_weights),
        }
    )
    execution_axis_id = f"{identity.axis_id}:economic:{axis_basis}"
    cost_rate: float | torch.Tensor
    if isinstance(full_bound.cost_rate, torch.Tensor) and full_bound.cost_rate.ndim > 0:
        cost_rate = full_bound.cost_rates[execution_start:]
    else:
        cost_rate = full_bound.cost_rate
    economic_sequence = Hold30Sequence(
        decision_state=full_bound.decision_state[execution_start:],
        asset_returns=full_bound.asset_returns[execution_start:],
        decision_available=full_bound.decision_available[execution_start:],
        fill_membership=full_bound.fill_membership[execution_start:],
        fill_availability=full_bound.fill_availability[execution_start:],
        benchmark_weights=full_bound.benchmark_weights[execution_start:],
        risk_asset_caps=full_bound.risk_asset_caps[execution_start:],
        risk_gross_max=full_bound.risk_gross_max[execution_start:],
        benchmark_net_returns=full_bound.benchmark_net_returns[execution_start:],
        initial_ledger=initial_ledger,
        cost_rate=cost_rate,
        initial_equity=full_bound.asset_returns.new_ones((full_bound.batch_size,)),
        track_entry_units=(
            None
            if full_bound.track_entry_units is None
            else full_bound.track_entry_units[execution_start:]
        ),
        axis_id=execution_axis_id,
    )
    receipt = Top2000M03RV72026EconomicExecutionReceipt(
        chronology_receipt_sha256=identity.receipt_sha256,
        pre2026_cache_sha256=pre2026_cache.cache_sha256,
        pre2026_cache_identity=pre2026_cache.cache_identity,
        training_fold_receipt_sha256=training_fold.receipt_sha256,
        training_fold_index=training_fold.fold_index,
        training_state_stop_exclusive=training_fold.training_state_stop_exclusive,
        training_cutoff_state_index=cutoff_index,
        training_cutoff_date=pre2026_cache.exchange_dates[cutoff_index],
        selected_context_cache_start_index=selected_cache_start,
        selected_context_start_date=retrospective.exchange_dates[0],
        economic_execution_start=execution_start,
        economic_execution_cache_state_index=execution_cache_index,
        economic_execution_decision_date=retrospective.exchange_dates[execution_start],
        first_economic_return_date=retrospective.exchange_dates[execution_start + 1],
        global_score_transition_start=identity.score_transition_start,
        global_score_transition_stop_exclusive=identity.score_transition_stop_exclusive,
        local_score_transition_start=identity.score_transition_start - execution_start,
        local_score_transition_stop_exclusive=(
            identity.score_transition_stop_exclusive - execution_start
        ),
        encoder_input_state_rows=identity.state_rows,
        encoder_only_economic_prefix_transition_rows=execution_start,
        executed_state_rows=economic_sequence.n_positions,
        executed_transition_rows=economic_sequence.n_positions - 1,
        initial_c1_weights_sha256=_tensor_sha256(initial_weights),
        initial_ledger_economic_value_sha256=_tensor_sha256(
            initial_ledger.economic_value
        ),
        initial_ledger_retention_units_sha256=_tensor_sha256(
            initial_ledger.retention_units
        ),
        execution_axis_id=execution_axis_id,
    )
    provider = _EncoderHistoryEconomicSuffixProvider(
        base=base_provider,
        full_sequence=full_bound,
        economic_execution_start=execution_start,
    )
    return Top2000M03RV72026EconomicExecutionView(
        sequence=economic_sequence,
        state_provider=cast(Hold30DecisionStateProvider, provider),
        receipt=receipt,
    )


__all__ = [
    "TOP2000_M03R_V7_2026_EXECUTION_VIEW_SCHEMA",
    "Top2000M03RV72026EconomicExecutionReceipt",
    "Top2000M03RV72026EconomicExecutionView",
    "Top2000M03RV72026ExecutionViewError",
    "build_top2000_m03r_v7_2026_economic_execution_view",
]
