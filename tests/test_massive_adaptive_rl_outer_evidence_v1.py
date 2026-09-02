from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from rl_quant.evaluation.massive_adaptive_rl_outer_evidence_authority_v1 import (
    MassiveAdaptiveRLOuterEvidenceAuthorityV1Error,
    authorize_massive_adaptive_rl_outer_evidence_authority_v1,
    materialize_massive_adaptive_rl_outer_evidence_authority_v1,
    parse_massive_adaptive_rl_outer_evidence_authority_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_evidence_v1 import (
    MASSIVE_ADAPTIVE_RL_OUTER_COST_FOLD_V1_SCHEMA,
    MassiveAdaptiveRLOuterCostFoldV1,
    MassiveAdaptiveRLOuterEvidenceV1Error,
    build_massive_adaptive_rl_outer_plan_v1,
    build_massive_adaptive_rl_outer_evidence_v1,
)
from rl_quant.execution.massive_adaptive_portfolio_compiler_v1 import (
    MassiveAdaptivePortfolioCompilerConfigV1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
)


def _fold(index: int) -> MassiveAdaptiveRLOuterCostFoldV1:
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_OUTER_COST_FOLD_V1_SCHEMA,
        "fold_index": index,
        "outer_plan_receipt_sha256": semantic_sha256(("outer-plan", index)),
        "frozen_rl_policy_receipt_sha256": semantic_sha256(
            ("frozen-policy", index)
        ),
        "low_cost_trace_receipt_sha256": semantic_sha256(("low", index)),
        "primary_trace_receipt_sha256": semantic_sha256(("primary", index)),
        "high_cost_trace_receipt_sha256": semantic_sha256(("high", index)),
        "best_fixed_control_trace_receipt_sha256": semantic_sha256(
            ("best-fixed-control", index)
        ),
        "decision_target_inventory_sha256": semantic_sha256(("targets", index)),
        "primary_strategy_active_log_returns": (0.0005,) * 126,
        "primary_incremental_rl_log_returns": (0.0002,) * 126,
        "primary_ppo_minus_fixed_control_log_returns": (0.0001,) * 126,
        "low_cost_terminal_return": 0.12,
        "primary_terminal_return": 0.08,
        "high_cost_terminal_return": 0.01,
        "maximum_drawdown": 0.05,
        "source_data_qualified": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    result = MassiveAdaptiveRLOuterCostFoldV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def test_rl_outer_evidence_is_paired_deterministic_and_create_only(tmp_path) -> None:
    folds = tuple(_fold(index) for index in range(4))
    first = build_massive_adaptive_rl_outer_evidence_v1(folds)
    second = build_massive_adaptive_rl_outer_evidence_v1(tuple(reversed(folds)))

    assert first.semantic_receipt_sha256 == second.semantic_receipt_sha256
    assert first.strategy_active_log_return_lcb95 == pytest.approx(0.0005)
    assert first.incremental_rl_log_return_lcb95 == pytest.approx(0.0002)
    assert first.ppo_minus_fixed_control_log_return_lcb95 == pytest.approx(
        0.0001
    )
    assert first.positive_strategy_fold_count == 4
    assert first.positive_incremental_fold_count == 4
    assert first.positive_ppo_minus_fixed_control_fold_count == 4
    assert not first.failed_gate_names
    assert not first.source_data_qualified
    assert not first.outer_development_conclusion_authorized
    assert not first.profitability_reporting_authorized
    assert not first.lockbox_access_authorized

    authority = materialize_massive_adaptive_rl_outer_evidence_authority_v1(
        root=tmp_path,
        artifact_id="synthetic-rl-outer-evidence",
        folds=folds,
        committed_at_ms=1,
    )
    assert authority.runtime_evidence_replayed
    assert authority.runtime_folds == folds
    assert not authority.outer_development_conclusion_authorized
    assert not authority.reinforcement_learning_authorized

    generic = parse_massive_adaptive_rl_outer_evidence_authority_v1(
        root=tmp_path,
        loaded_source=authority.loaded_source,
    )
    assert not generic.runtime_evidence_replayed
    assert generic.runtime_folds is None
    replayed = authorize_massive_adaptive_rl_outer_evidence_authority_v1(
        root=tmp_path,
        authority=generic,
        folds=tuple(reversed(folds)),
    )
    assert replayed.semantic_receipt_sha256 == authority.semantic_receipt_sha256


def test_rl_outer_evidence_preserves_nonmonotone_ladder_as_failed_gate(
    tmp_path,
) -> None:
    first_fold = replace(
        _fold(0),
        low_cost_terminal_return=0.07,
        semantic_receipt_sha256="0" * 64,
    )
    first_fold = replace(
        first_fold,
        semantic_receipt_sha256=semantic_sha256(first_fold.semantic_unsigned()),
    )
    first_fold.validate()
    assert not first_fold.terminal_return_ladder_monotone
    folds = (first_fold, *tuple(_fold(index) for index in range(1, 4)))

    evidence = build_massive_adaptive_rl_outer_evidence_v1(folds)

    assert not evidence.cost_ladder_monotone
    assert evidence.failed_gate_names == ("cost-ladder-monotone",)
    assert "cost-ladder-monotone" not in evidence.passed_gate_names
    authority = materialize_massive_adaptive_rl_outer_evidence_authority_v1(
        root=tmp_path,
        artifact_id="nonmonotone-rl-outer-evidence",
        folds=folds,
        committed_at_ms=1,
    )
    generic = parse_massive_adaptive_rl_outer_evidence_authority_v1(
        root=tmp_path,
        loaded_source=authority.loaded_source,
    )
    replayed = authorize_massive_adaptive_rl_outer_evidence_authority_v1(
        root=tmp_path,
        authority=generic,
        folds=folds,
    )
    assert replayed.runtime_evidence_replayed
    assert replayed.evidence.failed_gate_names == ("cost-ladder-monotone",)


def test_rl_outer_evidence_rejects_incomplete_or_mutated_folds(tmp_path) -> None:
    folds = tuple(_fold(index) for index in range(4))
    with pytest.raises(
        MassiveAdaptiveRLOuterEvidenceV1Error,
        match="exactly folds zero through three",
    ):
        build_massive_adaptive_rl_outer_evidence_v1(folds[:3])

    authority = materialize_massive_adaptive_rl_outer_evidence_authority_v1(
        root=tmp_path,
        artifact_id="mutated-rl-outer-evidence",
        folds=folds,
        committed_at_ms=2,
    )
    generic = parse_massive_adaptive_rl_outer_evidence_authority_v1(
        root=tmp_path,
        loaded_source=authority.loaded_source,
    )
    changed = replace(
        folds[0],
        primary_incremental_rl_log_returns=(0.0003,) * 126,
        semantic_receipt_sha256="0" * 64,
    )
    changed = replace(
        changed,
        semantic_receipt_sha256=semantic_sha256(changed.semantic_unsigned()),
    )
    with pytest.raises(
        MassiveAdaptiveRLOuterEvidenceAuthorityV1Error,
        match="does not replay",
    ):
        authorize_massive_adaptive_rl_outer_evidence_authority_v1(
            root=tmp_path,
            authority=generic,
            folds=(changed, *folds[1:]),
        )


def test_rl_outer_plan_binds_one_fold_checkpoint_calibration_and_policy() -> None:
    checkpoint_receipt = semantic_sha256("supervised-checkpoint")
    model_state_receipt = semantic_sha256("supervised-model-state")
    training_window_receipt = semantic_sha256("training-window")
    inference_receipt = semantic_sha256("outer-inference")
    selection_receipt = semantic_sha256("rl-policy-selection")
    selected_rl_checkpoint = semantic_sha256("selected-rl-checkpoint")
    policy_authority_receipt = semantic_sha256("rl-policy-authority")
    outer_inference = SimpleNamespace(
        validate=lambda: None,
        fold_index=0,
        semantic_receipt_sha256=inference_receipt,
        selected_checkpoint_receipt_sha256=checkpoint_receipt,
        outer_inference_authorized=True,
    )
    outer_forecasts = SimpleNamespace(
        validate=lambda: None,
        fold_index=0,
        semantic_receipt_sha256=semantic_sha256("outer-forecasts"),
        outer_inference_plan_receipt_sha256=inference_receipt,
        selected_checkpoint_receipt_sha256=checkpoint_receipt,
        model_state_receipt_sha256=model_state_receipt,
        training_window_plan_receipt_sha256=training_window_receipt,
        outer_forecast_authorized=True,
    )
    calibration = SimpleNamespace(
        validate=lambda: None,
        fold_index=0,
        semantic_receipt_sha256=semantic_sha256("calibration"),
        checkpoint_receipt_sha256=checkpoint_receipt,
        model_state_receipt_sha256=model_state_receipt,
        training_window_plan_receipt_sha256=training_window_receipt,
        development_calibration_authorized=True,
    )
    selection = SimpleNamespace(
        fold_index=0,
        semantic_receipt_sha256=selection_receipt,
        selected_checkpoint_receipt_sha256=selected_rl_checkpoint,
    )
    policy_selection_authority = SimpleNamespace(
        validate=lambda: None,
        semantic_receipt_sha256=policy_authority_receipt,
        runtime_selection=selection,
        runtime_selection_replayed=True,
        outer_evaluation_authorized=True,
    )
    frozen_policy = SimpleNamespace(
        validate=lambda: None,
        fold_index=0,
        policy_selection_authority_receipt_sha256=policy_authority_receipt,
        policy_selection_receipt_sha256=selection_receipt,
        selected_rl_checkpoint_receipt_sha256=selected_rl_checkpoint,
        semantic_receipt_sha256=semantic_sha256("frozen-policy"),
        frozen_model_state_receipt_sha256=semantic_sha256("rl-model-state"),
        observation_specification_sha256=semantic_sha256("observation"),
        action_specification_sha256=semantic_sha256("action"),
        reward_specification_sha256=semantic_sha256("reward"),
        development_outer_policy_authorized=True,
    )

    plan = build_massive_adaptive_rl_outer_plan_v1(
        outer_inference_plan=outer_inference,
        outer_forecast_archive=outer_forecasts,
        calibration=calibration,
        policy_selection_authority=policy_selection_authority,
        frozen_policy=frozen_policy,
        compiler_config=MassiveAdaptivePortfolioCompilerConfigV1(),
    )
    assert plan.fold_index == 0
    assert plan.outer_evaluation_authorized
    assert not plan.profitability_reporting_authorized
    assert not plan.lockbox_access_authorized

    frozen_policy.fold_index = 1
    with pytest.raises(
        MassiveAdaptiveRLOuterEvidenceV1Error,
        match="components differ",
    ):
        build_massive_adaptive_rl_outer_plan_v1(
            outer_inference_plan=outer_inference,
            outer_forecast_archive=outer_forecasts,
            calibration=calibration,
            policy_selection_authority=policy_selection_authority,
            frozen_policy=frozen_policy,
            compiler_config=MassiveAdaptivePortfolioCompilerConfigV1(),
        )
