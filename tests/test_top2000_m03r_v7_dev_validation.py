from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.evaluation.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_ENSEMBLE_RULE,
    Top2000M03RV7FoldEnsembleReceipt,
    Top2000M03RV7OutputSpaceEnsemblePolicy,
    Top2000M03RV7SeedValidationReceipt,
    Top2000M03RV7ValidationError,
    Top2000M03RV7ValidationTraceEvidence,
    aggregate_top2000_m03r_v7_intents,
    build_top2000_m03r_v7_validation_runtime,
    validate_fold_score_bounds,
)
from rl_quant.models.daily_policy import Hold30Intent
from rl_quant.protocol.hold30_alpha_m03r_v7_top2000_dev import (
    M03R_TOP2000_DEV_SETTING_IDS,
)
from rl_quant.training.hold30_runtime import TensorHold30DecisionStateProvider
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_SEEDS,
    Top2000M03RV7ActionBuilder,
    Top2000M03RV7DevelopmentPolicy,
    render_top2000_m03r_v7_development_folds,
)


def _digest(character: str) -> str:
    return character * 64


def _trace() -> Top2000M03RV7ValidationTraceEvidence:
    policy = torch.linspace(-0.001, 0.002, 63, dtype=torch.float64)
    benchmark = torch.linspace(-0.0005, 0.001, 63, dtype=torch.float64)
    sold = torch.zeros(61, dtype=torch.float64)
    sold[5] = 0.01
    sold[40] = 0.02
    terminal = torch.zeros(61, dtype=torch.float64)
    terminal[45] = 0.8
    discretionary = torch.full((63,), 0.01, dtype=torch.float64)
    forced = torch.full((63,), 0.002, dtype=torch.float64)
    return Top2000M03RV7ValidationTraceEvidence(
        policy_net_returns=policy,
        benchmark_net_returns=benchmark,
        active_log_returns=torch.log1p(policy) - torch.log1p(benchmark),
        total_one_way_turnover=discretionary + forced,
        discretionary_one_way_turnover=discretionary,
        forced_one_way_turnover=forced,
        discretionary_sold_notional_by_age=sold,
        terminal_risky_notional_by_age=terminal,
        score_transition_start=251,
        score_transition_stop_exclusive=314,
    )


def test_validation_trace_binds_exact_63_decisions_and_holding_metrics() -> None:
    trace = _trace()
    assert len(trace.trace_sha256) == 64
    assert set(trace.array_sha256s()) == {
        "policy_net_returns",
        "benchmark_net_returns",
        "active_log_returns",
        "total_one_way_turnover",
        "discretionary_one_way_turnover",
        "forced_one_way_turnover",
        "discretionary_sold_notional_by_age",
        "terminal_risky_notional_by_age",
    }
    metrics = trace.metrics()
    assert metrics["decision_count"] == 63
    assert metrics["discretionary_sold_notional"] == pytest.approx(0.03)
    assert metrics["median_discretionary_sale_age"] == 40
    assert metrics["terminal_notional_weighted_age"] == pytest.approx(45.0)

    with pytest.raises(Top2000M03RV7ValidationError, match="exactly 63"):
        replace(trace, score_transition_stop_exclusive=313)


def test_output_space_rule_aggregates_intents_not_return_paths() -> None:
    available = torch.tensor([[True, True, True]])
    intents = []
    for index in range(5):
        matrix = torch.tensor([[0.0, float(index), float(index + 2)]])
        vector = torch.tensor([0.01 + 0.001 * index])
        intents.append(
            Hold30Intent(
                entry_scores=matrix,
                hazard_residual=matrix,
                raw_hazard_residual=matrix,
                exposure_residual=torch.zeros(1),
                active_risk_scale=vector,
                signal_confidence=torch.tensor([0.5]),
                uncalibrated_signal_confidence_logit=torch.zeros(1),
            )
        )
    aggregate = aggregate_top2000_m03r_v7_intents(intents, available)
    assert aggregate.entry_scores is not None
    assert aggregate.entry_scores[0, 1].item() == pytest.approx(2.0)
    assert aggregate.active_risk_scale is not None
    assert aggregate.active_risk_scale.item() == pytest.approx(0.012)
    assert aggregate.raw_hazard_residual is not None
    assert aggregate.raw_hazard_residual[0, 1].item() == pytest.approx(2.0)
    assert aggregate.hazard_residual is not None
    assert aggregate.hazard_residual[0, 0].item() == -12.0


def test_validation_runtime_uses_the_exact_training_action_builder() -> None:
    policy = Top2000M03RV7DevelopmentPolicy(
        M03R_TOP2000_DEV_SETTING_IDS[0],
        token_dim=16,
        raw_stock_chunk=32,
    )
    runtime = build_top2000_m03r_v7_validation_runtime(
        policy,
        state_provider=TensorHold30DecisionStateProvider(),
    )
    assert isinstance(runtime.action_builder, Top2000M03RV7ActionBuilder)
    assert runtime.action_builder.policy is policy
    assert runtime.alpha_total_risk_step is None


def test_ensemble_binds_the_same_factor_execution_tensors() -> None:
    members = tuple(
        Top2000M03RV7DevelopmentPolicy(
            M03R_TOP2000_DEV_SETTING_IDS[0],
            token_dim=16,
            raw_stock_chunk=32,
        )
        for _ in range(5)
    )
    ensemble = Top2000M03RV7OutputSpaceEnsemblePolicy(members)
    loadings = torch.randn((121, 4), dtype=torch.float32)
    loadings[0] = 0.0
    ensemble.bind_episode_factor_loadings(loadings)
    assert torch.equal(ensemble.episode_factor_loadings, loadings)
    assert torch.equal(
        ensemble.episode_factor_constraint_pinv,
        members[0].episode_factor_constraint_pinv,
    )
    runtime = build_top2000_m03r_v7_validation_runtime(
        ensemble,
        state_provider=TensorHold30DecisionStateProvider(),
    )
    assert isinstance(runtime.action_builder, Top2000M03RV7ActionBuilder)
    assert runtime.action_builder.policy is ensemble


def test_fold_bounds_map_global_window_to_local_251_through_313() -> None:
    fold = render_top2000_m03r_v7_development_folds(1001)[0]
    sequence_start = fold.validation_decision_start - 251
    start, stop = validate_fold_score_bounds(
        fold,
        sequence_global_state_start=sequence_start,
        sequence_state_rows=378,
    )
    assert (start, stop) == (251, 314)
    assert sequence_start + start == fold.validation_decision_start
    assert sequence_start + stop == fold.validation_decision_stop_exclusive


def test_seed_and_fold_receipts_stay_development_only() -> None:
    trace = _trace()
    setting_id = M03R_TOP2000_DEV_SETTING_IDS[0]
    seed = Top2000M03RV7SeedValidationReceipt(
        setting_index=0,
        setting_id=setting_id,
        fold_index=0,
        seed=TOP2000_M03R_V7_DEV_SEEDS[0],
        fold_receipt_sha256=_digest("1"),
        sequence_receipt_sha256=_digest("2"),
        checkpoint_file_sha256=_digest("3"),
        model_state_sha256=_digest("4"),
        validation_trace_artifact_sha256=_digest("5"),
        validation_trace_sha256=trace.trace_sha256,
        array_sha256=trace.array_sha256s(),
        metrics=trace.metrics(),
        validation_global_decision_start=408,
        validation_global_decision_stop_exclusive=471,
        first_validation_date="2023-01-03",
        last_validation_date="2023-04-03",
    )
    assert len(seed.receipt_sha256) == 64
    with pytest.raises(Top2000M03RV7ValidationError, match="development-only"):
        replace(seed, promotion_eligible=True)

    fold = Top2000M03RV7FoldEnsembleReceipt(
        setting_index=0,
        setting_id=setting_id,
        fold_index=0,
        fold_receipt_sha256=_digest("1"),
        ordered_seeds=TOP2000_M03R_V7_DEV_SEEDS,
        seed_validation_receipt_sha256s=tuple(_digest(str(i)) for i in range(1, 6)),
        member_checkpoint_file_sha256s=tuple(_digest(str(i)) for i in range(1, 6)),
        member_model_state_sha256s=tuple(_digest(str(i)) for i in range(1, 6)),
        sequence_receipt_sha256=_digest("6"),
        validation_trace_artifact_sha256=_digest("7"),
        validation_trace_sha256=trace.trace_sha256,
        array_sha256=trace.array_sha256s(),
        metrics=trace.metrics(),
        validation_global_decision_start=408,
        validation_global_decision_stop_exclusive=471,
        first_validation_date="2023-01-03",
        last_validation_date="2023-04-03",
    )
    assert fold.ensemble_rule == TOP2000_M03R_V7_ENSEMBLE_RULE
    assert fold.seeds_are_independent_return_paths is False
    with pytest.raises(Top2000M03RV7ValidationError, match="five members"):
        replace(fold, seed_validation_receipt_sha256s=(_digest("1"),) * 4)
