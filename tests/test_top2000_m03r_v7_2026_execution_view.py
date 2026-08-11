from __future__ import annotations

import datetime as dt
import hashlib
import json

import pytest
import torch

from rl_quant.evaluation.top2000_m03r_v7_2026_execution_view import (
    build_top2000_m03r_v7_2026_economic_execution_view,
)
from rl_quant.evaluation.top2000_m03r_v7_2026_trace_telemetry import (
    adapt_top2000_m03r_v7_2026_trace,
)
from rl_quant.evaluation.top2000_m03r_v7_2026_retrospective_data import (
    TOP2000_M03R_V7_2026_RETROSPECTIVE_ACK,
    Top2000M03RV72026RetrospectiveData,
    Top2000M03RV72026RetrospectiveSourceEvidence,
    compose_top2000_m03r_v7_2026_retrospective_data,
)
from rl_quant.models.daily_policy import Hold30Intent
from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_dev import (
    M03R_SEED17_TOP2000_SETTING_IDS,
)
from rl_quant.training.hold30 import Hold30ReplayGeometry
from rl_quant.training.hold30_runtime import Hold30ChronologicalRuntime
from rl_quant.training.hold30_top2000_development import (
    DEVELOPMENT_ACK,
    Top2000VerifiedDevelopmentCache,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    Top2000M03RV7DevelopmentError,
    Top2000M03RV7DevelopmentPolicy,
    bind_top2000_m03r_v7_runtime_sequence,
    render_top2000_m03r_v7_development_folds,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _axis_digest(values: tuple[str, ...]) -> str:
    encoded = (
        json.dumps(
            list(values),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _weekdays(count: int, stop: dt.date) -> tuple[str, ...]:
    result: list[str] = []
    current = stop
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current.isoformat())
        current -= dt.timedelta(days=1)
    return tuple(reversed(result))


def _bars(rows: int, assets: int) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.zeros((rows, assets, 5), dtype=torch.float64)
    closes = 100.0 + torch.arange(rows, dtype=torch.float64) * 0.01
    values[:, 1:, 0] = closes[:, None]
    values[:, 1:, 1] = closes[:, None] + 1.0
    values[:, 1:, 2] = closes[:, None] - 1.0
    values[:, 1:, 3] = closes[:, None]
    values[:, 1:, 4] = 1_000_000.0
    return values, torch.ones((rows, assets), dtype=torch.bool)


def _cache_and_retrospective() -> tuple[
    Top2000VerifiedDevelopmentCache,
    Top2000M03RV72026RetrospectiveData,
]:
    actions = ("CASH", "A1", "A2", "A3", "A4")
    dates = _weekdays(1001, dt.date(2025, 12, 29))
    bars, available = _bars(len(dates), len(actions))
    cache = Top2000VerifiedDevelopmentCache(
        daily_ohlcv=bars,
        availability=available,
        exchange_dates=dates,
        action_ids=actions,
        cache_sha256=_digest("cache-file"),
        cache_identity=_digest("cache-identity"),
        search_identity=_digest("search"),
        action_hash=_axis_digest(actions),
        bar_seconds=300,
        acknowledgement=DEVELOPMENT_ACK,
        development_only=True,
        bars_only=True,
    )
    raw_dates = (
        dates[-1],
        "2026-01-02",
        "2026-01-05",
        "2026-06-22",
        "2026-06-23",
    )
    raw_bars, raw_available = _bars(len(raw_dates), len(actions))
    raw_bars[0] = bars[-1]
    source = Top2000M03RV72026RetrospectiveSourceEvidence(
        base_dataset_identity=_digest("base"),
        search_identity=cache.search_identity,
        lockbox_partition_names_hash=_digest("lockbox"),
        test_identity=_digest("test"),
        test_partition_inventory_sha256=_digest("partitions"),
        manifest_sha256=_digest("manifest"),
        universe_sha256=_digest("universe"),
        training_completion_receipt_sha256=_digest("complete"),
        evaluation_contract_sha256=_digest("contract"),
        raw_first_exchange_date=raw_dates[0],
        raw_last_exchange_date=raw_dates[-1],
    )
    retrospective = compose_top2000_m03r_v7_2026_retrospective_data(
        cache,
        retrospective_daily_ohlcv=raw_bars,
        retrospective_availability=raw_available,
        retrospective_exchange_dates=raw_dates,
        retrospective_action_ids=actions,
        source_evidence=source,
        acknowledgement=TOP2000_M03R_V7_2026_RETROSPECTIVE_ACK,
    )
    return cache, retrospective


class _EncoderPolicy(torch.nn.Module):
    state_provider_compatibility_id = (
        Top2000M03RV7DevelopmentPolicy.state_provider_compatibility_id
    )
    token_dim = 3

    def __init__(self) -> None:
        super().__init__()
        self.encoded_rows: list[int] = []

    def encode_episode(self, *args: object) -> torch.Tensor:
        availability = args[-1]
        assert isinstance(availability, torch.Tensor)
        batch, rows, assets = availability.shape
        self.encoded_rows.append(rows)
        return torch.zeros((batch, rows, assets, self.token_dim), dtype=torch.float64)

    def hold30_intent(
        self,
        state_t: torch.Tensor,
        prev_weights: torch.Tensor,
        available: torch.Tensor,
        age_summaries: torch.Tensor | None = None,
    ) -> Hold30Intent:
        del state_t, available, age_summaries
        return Hold30Intent(
            entry_scores=torch.zeros_like(prev_weights),
            hazard_residual=torch.zeros_like(prev_weights),
            exposure_residual=prev_weights.new_zeros(prev_weights.shape[0]),
        )


def test_fold5_uses_full_encoder_history_but_starts_economics_at_oos_return() -> None:
    cache, retrospective = _cache_and_retrospective()
    policy = _EncoderPolicy()
    fold = render_top2000_m03r_v7_development_folds(1001)[5]
    view = build_top2000_m03r_v7_2026_economic_execution_view(
        retrospective,
        cache,
        fold,
        policy,  # type: ignore[arg-type]
    )

    # The selected encoder context starts at cache index 749.  Fold 5 trained
    # through state 842, so its first admissible OOS return starts at state 842.
    assert view.receipt.training_cutoff_state_index == 842
    assert view.receipt.economic_execution_cache_state_index == 842
    assert view.receipt.economic_execution_start == 93
    assert view.receipt.local_score_transition_start == 251 - 93
    assert view.receipt.learned_policy_actions_before_execution_start == 0
    assert not view.receipt.in_sample_origin_holdings_enter_2026
    assert view.sequence.n_positions == retrospective.sequence.n_positions - 93
    assert torch.equal(
        view.sequence.initial_ledger.weights,
        retrospective.sequence.benchmark_weights[93],
    )
    assert not bool((view.sequence.initial_ledger.economic_value[..., 1:] != 0).any())
    assert not bool((view.sequence.initial_ledger.retention_units != 0).any())

    states = view.state_provider.canonical_states(policy, view.sequence)
    assert isinstance(states, torch.Tensor)
    assert states.shape[0] == view.sequence.n_positions - 1
    assert policy.encoded_rows == [retrospective.sequence.n_positions]


def test_earlier_fold_never_executes_a_policy_action_before_selected_context() -> None:
    cache, retrospective = _cache_and_retrospective()
    policy = _EncoderPolicy()
    fold = render_top2000_m03r_v7_development_folds(1001)[0]
    view = build_top2000_m03r_v7_2026_economic_execution_view(
        retrospective,
        cache,
        fold,
        policy,  # type: ignore[arg-type]
    )

    assert view.receipt.training_cutoff_state_index == 377
    assert view.receipt.economic_execution_start == 0
    assert view.receipt.economic_execution_cache_state_index == 749
    assert view.receipt.first_economic_return_date > view.receipt.training_cutoff_date
    assert view.receipt.local_score_transition_start == 251


def test_offset_trace_runs_once_without_padding_and_maps_global_score_rows() -> None:
    cache, retrospective = _cache_and_retrospective()
    policy = _EncoderPolicy()
    fold = render_top2000_m03r_v7_development_folds(1001)[5]
    view = build_top2000_m03r_v7_2026_economic_execution_view(
        retrospective,
        cache,
        fold,
        policy,  # type: ignore[arg-type]
    )
    runtime = Hold30ChronologicalRuntime(
        "H2",
        state_provider=view.state_provider,
        require_trainable_state_provider=False,
    )
    roles = Hold30ReplayGeometry(
        warmup_decisions=63,
        label_support_decisions=63,
        max_origin_batch=1,
    ).roles(view.sequence.n_positions)
    with torch.no_grad():
        trace, _rows = runtime.canonical_pass(policy, view.sequence, roles)

    assert len(trace.transitions) == retrospective.identity.transition_rows - 93
    result = adapt_top2000_m03r_v7_2026_trace(
        trace,
        retrospective,
        setting_id=M03R_SEED17_TOP2000_SETTING_IDS[3],
        checkpoint_sha256=_digest("checkpoint"),
        checkpoint_fold_index=5,
        economic_execution_view=view,
    )
    assert result.receipt.economic_execution_start == 93
    assert result.receipt.score_transition_start == 251 - 93
    assert result.receipt.global_score_transition_start == 251
    assert result.receipt.completed_transition_rows == len(trace.transitions)
    assert result.score_dates == retrospective.score_return_dates


def test_compact_placeholder_preserves_provider_states_and_runtime_trace() -> None:
    _cache, retrospective = _cache_and_retrospective()
    policy = _EncoderPolicy()
    default_sequence, default_provider = bind_top2000_m03r_v7_runtime_sequence(
        retrospective.sequence,
        policy,  # type: ignore[arg-type]
    )
    compact_sequence, compact_provider = bind_top2000_m03r_v7_runtime_sequence(
        retrospective.sequence,
        policy,  # type: ignore[arg-type]
        placeholder_token_dim=1,
    )
    with pytest.raises(Top2000M03RV7DevelopmentError, match="positive integer"):
        bind_top2000_m03r_v7_runtime_sequence(
            retrospective.sequence,
            policy,  # type: ignore[arg-type]
            placeholder_token_dim=0,
        )
    assert default_sequence.decision_state.shape[-1] == policy.token_dim
    assert compact_sequence.decision_state.shape[-1] == 1
    assert default_sequence.decision_state.numel() == (
        compact_sequence.decision_state.numel() * policy.token_dim
    )
    with torch.no_grad():
        default_states = default_provider.canonical_states(policy, default_sequence)
        compact_states = compact_provider.canonical_states(policy, compact_sequence)
    assert torch.equal(default_states, compact_states)

    roles = Hold30ReplayGeometry().roles(default_sequence.n_positions)
    default_runtime = Hold30ChronologicalRuntime(
        "H2",
        state_provider=default_provider,
        require_trainable_state_provider=False,
    )
    compact_runtime = Hold30ChronologicalRuntime(
        "H2",
        state_provider=compact_provider,
        require_trainable_state_provider=False,
    )
    with torch.no_grad():
        default_trace, _ = default_runtime.canonical_pass(
            policy,
            default_sequence,
            roles,
        )
        compact_trace, _ = compact_runtime.canonical_pass(
            policy,
            compact_sequence,
            roles,
        )
    assert len(default_trace.transitions) == len(compact_trace.transitions)
    assert all(
        torch.equal(left.net_return, right.net_return)
        and torch.equal(left.filled_delta, right.filled_delta)
        for left, right in zip(
            default_trace.transitions,
            compact_trace.transitions,
            strict=True,
        )
    )
