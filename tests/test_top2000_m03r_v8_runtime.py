from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.execution.top2000_m03r_v8_projection import (
    qualify_m03r_v8_risk_manifest,
)
from rl_quant.models.daily_policy import Hold30Intent
from rl_quant.protocol.hold30_alpha_m03r_v8_top2000_dev import (
    resolve_m03r_v8_top2000_dev_setting,
)
from rl_quant.training.hold30_runtime import Hold30ChronologicalRuntime, Hold30Sequence
from rl_quant.training.top2000_m03r_v8_policy import (
    Top2000M03RV8DevelopmentPolicy,
)
from rl_quant.training.top2000_m03r_v8_runtime import (
    M03RV8RuntimeError,
    Top2000M03RV8ActionBuilder,
    build_top2000_m03r_v8_chronological_runtime,
)


def _books() -> tuple[
    CohortLedger, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    weights = torch.tensor(
        [[0.96, 0.008, 0.008, 0.008, 0.008, 0.008]],
        dtype=torch.float64,
    )
    ledger = CohortLedger.from_weights(
        weights,
        cash_index=0,
        initial_age=20,
        track_initial_units=True,
    )
    benchmark = weights.clone()
    trade_mask = torch.ones_like(weights, dtype=torch.bool)
    caps = torch.tensor(
        [[1.0, 0.01, 0.01, 0.01, 0.01, 0.01]],
        dtype=torch.float64,
    )
    gross = torch.ones(1, dtype=torch.float64)
    return ledger, benchmark, trade_mask, caps, gross


def _intent(
    *,
    confidence: float = 0.8,
    hazard: float = -12.0,
) -> Hold30Intent:
    alpha = torch.tensor(
        [[0.0, 0.008, 0.004, -0.004, -0.008, 0.002]],
        dtype=torch.float64,
    )
    return Hold30Intent(
        entry_scores=alpha,
        hazard_residual=torch.full_like(alpha, hazard),
        raw_hazard_residual=torch.full_like(alpha, hazard),
        exposure_residual=torch.zeros(1, dtype=torch.float64),
        alpha_mean_30d=alpha,
        alpha_downside_30d=torch.full_like(alpha, 0.002),
        active_risk_scale=torch.tensor([0.04], dtype=torch.float64),
        signal_confidence=torch.tensor([confidence], dtype=torch.float64),
        uncalibrated_signal_confidence_logit=torch.zeros(1, dtype=torch.float64),
        benchmark_derisk_request=torch.zeros(1, dtype=torch.float64),
        auxiliary_alpha_mean=torch.stack((alpha, alpha, alpha, alpha), dim=-1),
    )


def _builder(setting_index: int = 0) -> Top2000M03RV8ActionBuilder:
    # CASH is zero exposure and the benchmark is neutral by construction.
    loadings = torch.tensor(
        [[0.0], [1.0], [-1.0], [0.5], [-0.5], [0.0]],
        dtype=torch.float64,
    )
    covariance = torch.eye(6, dtype=torch.float64) * 1.0e-6
    covariance[0, 0] = 0.0
    risk_manifest = qualify_m03r_v8_risk_manifest(
        exposure_names=("synthetic-factor",),
        asset_axis_sha256="a" * 64,
        source_receipt_sha256="b" * 64,
        exposure_loadings=loadings,
        exposure_lower_bounds=torch.tensor([-0.0002], dtype=torch.float64),
        exposure_upper_bounds=torch.tensor([0.0002], dtype=torch.float64),
        active_beta_loadings=torch.zeros(6, dtype=torch.float64),
        daily_return_covariance=covariance,
        cash_index=0,
    )
    return Top2000M03RV8ActionBuilder(
        resolve_m03r_v8_top2000_dev_setting(setting_index),
        risk_manifest,
    )


def test_zero_confidence_preserves_the_independently_projected_hazard_anchor() -> None:
    ledger, benchmark, trade_mask, caps, gross = _books()
    built, trace = _builder().build_with_trace(
        _intent(confidence=0.0, hazard=4.0),
        ledger,
        benchmark,
        trade_mask,
        caps,
        gross,
    )

    assert not torch.equal(trace.raw_hazard_anchor_weights, ledger.weights)
    assert torch.equal(trace.gated_proposal_weights, trace.hazard_anchor_weights)
    assert torch.equal(trace.projected_weights, trace.hazard_anchor_weights)
    assert torch.equal(built.target_weights, trace.executed_weights)
    assert trace.proposed_release.sum() > 0.0
    trace.validate()


def test_cost_gate_modes_are_distinct_before_projection() -> None:
    ledger, benchmark, trade_mask, caps, gross = _books()
    traces = []
    for setting_index in (0, 4, 5):
        _built, trace = _builder(setting_index).build_with_trace(
            _intent(),
            ledger,
            benchmark,
            trade_mask,
            caps,
            gross,
        )
        traces.append(trace)

    reference, disabled, strong = traces
    assert disabled.proposal.requested_incremental_one_way_turnover > (
        reference.proposal.requested_incremental_one_way_turnover
    )
    assert reference.proposal.requested_incremental_one_way_turnover > (
        strong.proposal.requested_incremental_one_way_turnover
    )
    assert not torch.equal(
        reference.gated_proposal_weights,
        disabled.gated_proposal_weights,
    )
    assert not torch.equal(
        reference.gated_proposal_weights, strong.gated_proposal_weights
    )


def test_nonzero_bound_manifest_makes_the_relaxed_factor_row_causal() -> None:
    ledger, benchmark, trade_mask, caps, gross = _books()
    _reference_built, reference = _builder(0).build_with_trace(
        _intent(), ledger, benchmark, trade_mask, caps, gross
    )
    _relaxed_built, relaxed = _builder(7).build_with_trace(
        _intent(), ledger, benchmark, trade_mask, caps, gross
    )

    assert relaxed.proposal_projection.radial_scale > (
        reference.proposal_projection.radial_scale
    )
    assert not torch.equal(relaxed.projected_weights, reference.projected_weights)


def test_trace_receipt_binds_each_economic_book_and_rejects_mutation() -> None:
    ledger, benchmark, trade_mask, caps, gross = _books()
    _built, trace = _builder().build_with_trace(
        _intent(),
        ledger,
        benchmark,
        trade_mask,
        caps,
        gross,
    )
    receipt = trace.receipt_payload

    assert len(receipt["receipt_sha256"]) == 64
    assert set(receipt["book_sha256"]) == {
        "repaired_weights",
        "raw_hazard_anchor_weights",
        "hazard_anchor_weights",
        "gated_proposal_weights",
        "projected_weights",
        "executed_weights",
        "proposed_release_by_age",
        "proposed_release",
    }
    with pytest.raises(M03RV8RuntimeError, match="executed book"):
        replace(
            trace,
            executed_weights=trace.executed_weights
            + torch.tensor(
                [[0.0, 0.001, -0.001, 0.0, 0.0, 0.0]],
                dtype=torch.float64,
            ),
        ).validate()


class _Policy:
    def __init__(self, intent: Hold30Intent) -> None:
        self.intent = intent

    def hold30_intent(
        self,
        state_t: torch.Tensor,
        prev_weights: torch.Tensor,
        available: torch.Tensor,
        age_summaries: torch.Tensor | None = None,
    ) -> Hold30Intent:
        del state_t, prev_weights, available, age_summaries
        return self.intent


def test_builder_runs_inside_the_authoritative_delayed_fill_runtime() -> None:
    ledger, benchmark, trade_mask, caps, gross = _books()
    sequence = Hold30Sequence(
        decision_state=torch.zeros((2, 1, 6, 1), dtype=torch.float64),
        asset_returns=torch.zeros((1, 1, 6), dtype=torch.float64),
        decision_available=trade_mask.unsqueeze(0).expand(2, -1, -1).clone(),
        fill_membership=trade_mask.unsqueeze(0).expand(2, -1, -1).clone(),
        fill_availability=trade_mask.unsqueeze(0).expand(2, -1, -1).clone(),
        benchmark_weights=benchmark.unsqueeze(0).expand(2, -1, -1).clone(),
        risk_asset_caps=caps.unsqueeze(0).expand(2, -1, -1).clone(),
        risk_gross_max=gross.unsqueeze(0).expand(2, -1).clone(),
        benchmark_net_returns=torch.zeros((1, 1), dtype=torch.float64),
        initial_ledger=ledger,
        cost_rate=0.002,
        axis_id="v8-runtime-test",
    )
    runtime = Hold30ChronologicalRuntime("H2", action_builder=_builder())
    state = runtime.initial_state(sequence)
    state = runtime.decide(_Policy(_intent()), sequence, state)
    next_state, transition = runtime.advance(sequence, state)

    assert next_state.position_index == 1
    assert transition.fill_index == 1
    assert torch.equal(transition.pre_cost_weights, transition.post_cost_weights)
    assert transition.cost.item() >= 0.0
    assert torch.isfinite(transition.utility).all()


def test_policy_factory_binds_the_same_setting_to_the_delayed_fill_builder() -> None:
    policy = Top2000M03RV8DevelopmentPolicy(
        7,
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )
    risk_manifest = _builder(7).risk_manifest
    runtime = build_top2000_m03r_v8_chronological_runtime(policy, risk_manifest)

    assert runtime.mechanism == "H2"
    assert isinstance(runtime.action_builder, Top2000M03RV8ActionBuilder)
    assert runtime.action_builder.setting == policy.setting
    assert runtime.action_builder.risk_manifest is risk_manifest


def test_builder_rejects_missing_distributional_or_confidence_outputs() -> None:
    ledger, benchmark, trade_mask, caps, gross = _books()
    with pytest.raises(M03RV8RuntimeError, match="mean, uncertainty, and confidence"):
        _builder().build_with_trace(
            replace(_intent(), alpha_downside_30d=None),
            ledger,
            benchmark,
            trade_mask,
            caps,
            gross,
        )
