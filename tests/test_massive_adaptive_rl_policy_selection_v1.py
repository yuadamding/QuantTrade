from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from rl_quant.evaluation.massive_adaptive_rl_outer_plan_v2 import (
    MassiveAdaptiveRLOuterPlanV2Error,
    build_massive_adaptive_rl_outer_plan_v2,
)
from rl_quant.evaluation.massive_adaptive_rl_fixed_control_evaluator_v1 import (
    evaluate_massive_adaptive_rl_fixed_control_v1,
)
from rl_quant.rl.massive_adaptive_ppo_policy_v1 import (
    MassiveAdaptivePPOActorCriticV1,
)
from rl_quant.rl.massive_adaptive_rl_action_v1 import (
    neutral_massive_adaptive_rl_action_v1,
)
from rl_quant.training.massive_adaptive_frozen_rl_policy_v1 import (
    authorize_massive_adaptive_frozen_rl_policy_v1,
    materialize_massive_adaptive_frozen_rl_policy_v1,
    parse_massive_adaptive_frozen_rl_policy_v1,
)
from rl_quant.training.massive_adaptive_ppo_v1 import (
    MassiveAdaptivePPOTrainerV1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_selection_v1 import (
    build_massive_adaptive_rl_fixed_control_candidate_v1,
    materialize_massive_adaptive_rl_fixed_control_selection_authority_v1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_registry_v1 import (
    build_massive_adaptive_rl_fixed_control_registry_v1,
    registered_massive_adaptive_rl_constant_actions_v1,
    validate_massive_adaptive_rl_fixed_control_registry_coverage_v1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v1 import (
    MASSIVE_ADAPTIVE_RL_POLICY_TRACE_V1_SCHEMA,
    MassiveAdaptiveRLPolicySelectionV1Error,
    MassiveAdaptiveRLPolicyTraceV1,
    authorize_massive_adaptive_rl_policy_selection_authority_v1,
    build_massive_adaptive_rl_policy_candidate_v1,
    materialize_massive_adaptive_rl_policy_selection_authority_v1,
    parse_massive_adaptive_rl_policy_selection_authority_v1,
)
from test_massive_adaptive_profitability_v1_vertical_slice import (
    _adaptive_env_fixture,
)


def _digest(value: object) -> str:
    return semantic_sha256(value)


def _trace(
    *,
    cost: float,
    terminal_return: float,
    incremental: float,
    active: float,
    frozen: bool,
    role: str = "inner_validation",
) -> MassiveAdaptiveRLPolicyTraceV1:
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_POLICY_TRACE_V1_SCHEMA,
        "fold_index": 0,
        "evaluation_role": role,
        "checkpoint_receipt_sha256": _digest("rl-checkpoint"),
        "model_state_receipt_sha256": _digest("rl-model-state"),
        "update_index": 7,
        "training_forecast_authority_receipt_sha256": _digest(
            "rl-training-forecast-authority"
        ),
        "forecast_archive_receipt_sha256": _digest("validation-forecast"),
        "inference_plan_receipt_sha256": _digest("validation-plan"),
        "calibration_receipt_sha256": _digest("validation-calibration"),
        "transaction_cost_basis_points": cost,
        "initial_capital": 10_000_000.0,
        "decision_session_dates": ("2024-01-02", "2024-01-03"),
        "transition_receipts": (
            _digest((cost, "transition", 0)),
            _digest((cost, "transition", 1)),
        ),
        "decision_target_inventory_sha256": _digest("same-policy-targets"),
        "economic_source_inventory_sha256": _digest("same-economic-sources"),
        "strategy_active_log_returns": (active / 2.0, active / 2.0),
        "incremental_rl_log_returns": (incremental / 2.0, incremental / 2.0),
        "cumulative_strategy_active_log_return": active,
        "cumulative_incremental_rl_log_return": incremental,
        "terminal_liquidation_adjusted_return": terminal_return,
        "maximum_drawdown": 0.05,
        "frozen_targets_replayed": frozen,
        "source_data_qualified": False,
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    result = MassiveAdaptiveRLPolicyTraceV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def _checkpoint() -> SimpleNamespace:
    return SimpleNamespace(
        validate=lambda: None,
        semantic_receipt_sha256=_digest("rl-checkpoint"),
        model_state_receipt_sha256=_digest("rl-model-state"),
        update_index=7,
    )


def _fixed_control_authority(tmp_path):
    candidate = build_massive_adaptive_rl_fixed_control_candidate_v1(
        fold_index=0,
        control_id="C0-neutral",
        action=neutral_massive_adaptive_rl_action_v1(),
        training_trace=_trace(
            cost=20.0,
            terminal_return=0.01,
            incremental=0.004,
            active=0.01,
            frozen=False,
            role="training_control",
        ),
    )
    return materialize_massive_adaptive_rl_fixed_control_selection_authority_v1(
        root=tmp_path,
        artifact_id="fixed-control-selection",
        candidates=(candidate,),
        committed_at_ms=1,
    )


def _fixed_validation_trace() -> MassiveAdaptiveRLPolicyTraceV1:
    return _trace(
        cost=20.0,
        terminal_return=0.01,
        incremental=0.005,
        active=0.01,
        frozen=False,
    )


def _fit_authority_for_selection(*, registry, selection_authority, chronology):
    selection = selection_authority.runtime_selection
    candidates = selection_authority.runtime_candidates
    assert selection is not None
    assert candidates is not None
    fit_run = SimpleNamespace(
        fixed_control_registry_receipt_sha256=registry.semantic_receipt_sha256,
        chronology_authority_receipt_sha256=chronology.semantic_receipt_sha256,
        training_origin_inventory_sha256=(chronology.rl_fit_origin_inventory_sha256),
        candidate_inventory_sha256=selection.candidate_inventory_sha256,
        candidates=candidates,
    )
    return SimpleNamespace(
        validate=lambda: None,
        runtime_fit_run=fit_run,
        runtime_fit_replayed=True,
        development_control_fit_authorized=False,
        loaded_source=SimpleNamespace(
            commit=SimpleNamespace(committed_at_ms=1),
        ),
        semantic_receipt_sha256=_digest(
            ("fixed-control-fit-authority", selection.semantic_receipt_sha256)
        ),
    )


def test_fc06_is_fit_only_selection_from_complete_registered_grid(tmp_path) -> None:
    training_trace = _trace(
        cost=20.0,
        terminal_return=0.01,
        incremental=0.004,
        active=0.01,
        frozen=False,
        role="training_control",
    )
    candidates = tuple(
        build_massive_adaptive_rl_fixed_control_candidate_v1(
            fold_index=0,
            control_id=control_id,
            action=action,
            training_trace=training_trace,
        )
        for control_id, action in registered_massive_adaptive_rl_constant_actions_v1()
    )
    authority = materialize_massive_adaptive_rl_fixed_control_selection_authority_v1(
        root=tmp_path,
        artifact_id="complete-fc06-selection",
        candidates=candidates,
        committed_at_ms=2,
    )
    chronology = SimpleNamespace(
        validate=lambda: None,
        fold_index=0,
        rl_fit_origin_inventory_sha256=semantic_sha256(
            training_trace.decision_session_dates
        ),
        semantic_receipt_sha256=_digest("complete-fc06-chronology"),
    )
    registry = build_massive_adaptive_rl_fixed_control_registry_v1()
    fit_authority = _fit_authority_for_selection(
        registry=registry,
        selection_authority=authority,
        chronology=chronology,
    )
    validate_massive_adaptive_rl_fixed_control_registry_coverage_v1(
        registry=registry,
        fit_authority=fit_authority,  # type: ignore[arg-type]
        selection_authority=authority,
        chronology_authority=chronology,  # type: ignore[arg-type]
    )
    assert authority.runtime_selection is not None
    assert authority.runtime_selection.selected_control_id == "FC12"
    assert tuple(
        sorted(row.control_id for row in authority.runtime_candidates or ())
    ) == tuple(
        sorted(
            control_id
            for control_id, _action in (
                registered_massive_adaptive_rl_constant_actions_v1()
            )
        )
    )

    outer_plan_v1 = SimpleNamespace(
        validate=lambda: None,
        fold_index=0,
        semantic_receipt_sha256=_digest("outer-plan-v1"),
        outer_forecast_archive_receipt_sha256=_digest("outer-forecast-v1"),
        source_data_qualified=False,
        outer_evaluation_authorized=False,
    )
    outer_forecast = SimpleNamespace(
        validate=lambda: None,
        semantic_receipt_sha256=outer_plan_v1.outer_forecast_archive_receipt_sha256,
        loaded_source=SimpleNamespace(
            commit=SimpleNamespace(committed_at_ms=3),
        ),
    )
    bound_outer = build_massive_adaptive_rl_outer_plan_v2(
        outer_plan_v1=outer_plan_v1,  # type: ignore[arg-type]
        outer_forecast_archive=outer_forecast,  # type: ignore[arg-type]
        fixed_control_registry=registry,
        fixed_control_fit_authority=fit_authority,  # type: ignore[arg-type]
        fixed_control_selection_authority=authority,
        chronology_authority=chronology,  # type: ignore[arg-type]
    )
    assert bound_outer.selected_fixed_control_id == "FC12"
    assert bound_outer.comparator_frozen_at_ms == 2
    assert bound_outer.outer_forecast_committed_at_ms == 3

    outer_forecast.loaded_source.commit.committed_at_ms = 2
    with pytest.raises(MassiveAdaptiveRLOuterPlanV2Error, match="before outer"):
        build_massive_adaptive_rl_outer_plan_v2(
            outer_plan_v1=outer_plan_v1,  # type: ignore[arg-type]
            outer_forecast_archive=outer_forecast,  # type: ignore[arg-type]
            fixed_control_registry=registry,
            fixed_control_fit_authority=fit_authority,  # type: ignore[arg-type]
            fixed_control_selection_authority=authority,
            chronology_authority=chronology,  # type: ignore[arg-type]
        )


def test_fc06_validation_trace_is_generated_from_fit_selected_action(tmp_path) -> None:
    training_trace = _trace(
        cost=20.0,
        terminal_return=0.01,
        incremental=0.004,
        active=0.01,
        frozen=False,
        role="training_control",
    )
    training_trace = replace(
        training_trace,
        decision_session_dates=("2023-01-03", "2023-01-04"),
        transition_receipts=(_digest("fit-transition-0"), _digest("fit-transition-1")),
        semantic_receipt_sha256="0" * 64,
    )
    training_trace = replace(
        training_trace,
        semantic_receipt_sha256=semantic_sha256(training_trace.semantic_unsigned()),
    )
    candidates = tuple(
        build_massive_adaptive_rl_fixed_control_candidate_v1(
            fold_index=0,
            control_id=control_id,
            action=action,
            training_trace=training_trace,
        )
        for control_id, action in registered_massive_adaptive_rl_constant_actions_v1()
    )
    selection_authority = (
        materialize_massive_adaptive_rl_fixed_control_selection_authority_v1(
            root=tmp_path,
            artifact_id="fc06-package-evaluation",
            candidates=candidates,
            committed_at_ms=3,
        )
    )
    _, _, environment = _adaptive_env_fixture()
    validation_dates = tuple(
        row.decision_session_date for row in environment.inference_plan.rows
    )
    chronology = SimpleNamespace(
        validate=lambda: None,
        fold_index=0,
        training_forecast_authority_receipt_sha256=_digest(
            "rl-training-forecast-authority"
        ),
        rl_fit_origin_inventory_sha256=semantic_sha256(
            training_trace.decision_session_dates
        ),
        rl_validation_origin_dates=validation_dates,
        development_policy_selection_authorized=True,
        semantic_receipt_sha256=_digest("fc06-validation-chronology"),
    )
    registry = build_massive_adaptive_rl_fixed_control_registry_v1()
    fit_authority = _fit_authority_for_selection(
        registry=registry,
        selection_authority=selection_authority,
        chronology=chronology,
    )
    evaluation = evaluate_massive_adaptive_rl_fixed_control_v1(
        registry=registry,
        fit_authority=fit_authority,  # type: ignore[arg-type]
        selection_authority=selection_authority,
        chronology_authority=chronology,  # type: ignore[arg-type]
        environment=environment,
    )

    assert evaluation.policy_trace.decision_session_dates == validation_dates
    assert evaluation.transition_receipts == (
        evaluation.policy_trace.transition_receipts
    )
    assert len(evaluation.transition_receipts) == len(validation_dates)
    assert selection_authority.runtime_selection is not None
    assert evaluation.selected_action_receipt_sha256 == (
        selection_authority.runtime_selection.selected_action_receipt_sha256
    )
    assert not evaluation.development_policy_selection_authorized


def test_policy_selection_replays_frozen_cost_ladder_create_only(tmp_path) -> None:
    low = _trace(
        cost=10.0, terminal_return=0.12, incremental=0.025, active=0.04, frozen=True
    )
    primary = _trace(
        cost=20.0,
        terminal_return=0.08,
        incremental=0.02,
        active=0.03,
        frozen=False,
    )
    high = _trace(
        cost=40.0, terminal_return=0.01, incremental=0.01, active=0.02, frozen=True
    )
    candidate = build_massive_adaptive_rl_policy_candidate_v1(
        checkpoint=_checkpoint(),  # type: ignore[arg-type]
        primary_trace=primary,
        low_cost_trace=low,
        high_cost_trace=high,
        fixed_control_selection_authority=_fixed_control_authority(tmp_path),
        fixed_control_validation_trace=_fixed_validation_trace(),
    )

    assert candidate.economically_eligible
    authority = materialize_massive_adaptive_rl_policy_selection_authority_v1(
        root=tmp_path,
        artifact_id="rl-policy-selection",
        candidates=(candidate,),
        committed_at_ms=1,
    )
    assert authority.runtime_selection_replayed
    assert authority.runtime_selection is not None
    assert not authority.development_policy_selection_authorized
    assert not authority.outer_evaluation_authorized

    generic = parse_massive_adaptive_rl_policy_selection_authority_v1(
        root=tmp_path,
        loaded_source=authority.loaded_source,
    )
    assert generic.runtime_selection is None
    assert not generic.runtime_selection_replayed
    replayed = authorize_massive_adaptive_rl_policy_selection_authority_v1(
        root=tmp_path,
        authority=generic,
        candidates=(candidate,),
    )
    assert replayed.semantic_receipt_sha256 == authority.semantic_receipt_sha256

    changed = replace(
        candidate,
        primary_incremental_rl_log_wealth=0.03,
        semantic_receipt_sha256="0" * 64,
    )
    changed = replace(
        changed,
        semantic_receipt_sha256=semantic_sha256(changed.semantic_unsigned()),
    )
    with pytest.raises(
        MassiveAdaptiveRLPolicySelectionV1Error, match="does not replay"
    ):
        authorize_massive_adaptive_rl_policy_selection_authority_v1(
            root=tmp_path,
            authority=generic,
            candidates=(changed,),
        )


def test_policy_candidate_rejects_reoptimized_cost_stress(tmp_path) -> None:
    primary = _trace(
        cost=20.0,
        terminal_return=0.08,
        incremental=0.02,
        active=0.03,
        frozen=False,
    )
    low = replace(
        _trace(
            cost=10.0, terminal_return=0.12, incremental=0.025, active=0.04, frozen=True
        ),
        decision_target_inventory_sha256=_digest("different-targets"),
        semantic_receipt_sha256="0" * 64,
    )
    low = replace(low, semantic_receipt_sha256=semantic_sha256(low.semantic_unsigned()))
    with pytest.raises(MassiveAdaptiveRLPolicySelectionV1Error, match="cost ladder"):
        build_massive_adaptive_rl_policy_candidate_v1(
            checkpoint=_checkpoint(),  # type: ignore[arg-type]
            primary_trace=primary,
            low_cost_trace=low,
            high_cost_trace=_trace(
                cost=40.0,
                terminal_return=0.01,
                incremental=0.01,
                active=0.02,
                frozen=True,
            ),
            fixed_control_selection_authority=_fixed_control_authority(tmp_path),
            fixed_control_validation_trace=_fixed_validation_trace(),
        )


def test_policy_must_beat_training_selected_fixed_control(tmp_path) -> None:
    candidate = build_massive_adaptive_rl_policy_candidate_v1(
        checkpoint=_checkpoint(),  # type: ignore[arg-type]
        low_cost_trace=_trace(
            cost=10.0,
            terminal_return=0.03,
            incremental=0.004,
            active=0.02,
            frozen=True,
        ),
        primary_trace=_trace(
            cost=20.0,
            terminal_return=0.02,
            incremental=0.003,
            active=0.015,
            frozen=False,
        ),
        high_cost_trace=_trace(
            cost=40.0,
            terminal_return=0.001,
            incremental=0.001,
            active=0.005,
            frozen=True,
        ),
        fixed_control_selection_authority=_fixed_control_authority(tmp_path),
        fixed_control_validation_trace=_fixed_validation_trace(),
    )

    assert candidate.ppo_minus_best_fixed_control_log_wealth < 0.0
    assert "best-fixed-control" in candidate.eligibility_failures
    assert not candidate.economically_eligible


def test_selected_policy_model_state_is_create_only_and_exact(tmp_path) -> None:
    _, _, environment = _adaptive_env_fixture()
    environment.reset()
    trainer = MassiveAdaptivePPOTrainerV1(
        environment=environment,
        model=MassiveAdaptivePPOActorCriticV1(observation_dim=90),
    )
    checkpoint = trainer.checkpoint()
    training_authority_receipt = _digest("rl-training-forecast-authority")
    checkpoint = replace(
        checkpoint,
        training_forecast_authority_receipt_sha256=training_authority_receipt,
        development_rl_training_authorized=True,
        semantic_receipt_sha256="0" * 64,
    )
    checkpoint = replace(
        checkpoint,
        semantic_receipt_sha256=semantic_sha256(checkpoint.semantic_unsigned()),
    )
    checkpoint.validate()

    def trace(
        cost: float, terminal: float, *, frozen: bool
    ) -> MassiveAdaptiveRLPolicyTraceV1:
        value = _trace(
            cost=cost,
            terminal_return=terminal,
            incremental=0.02,
            active=0.03,
            frozen=frozen,
        )
        changed = replace(
            value,
            checkpoint_receipt_sha256=checkpoint.semantic_receipt_sha256,
            model_state_receipt_sha256=checkpoint.model_state_receipt_sha256,
            update_index=checkpoint.update_index,
            training_forecast_authority_receipt_sha256=training_authority_receipt,
            semantic_receipt_sha256="0" * 64,
        )
        return replace(
            changed,
            semantic_receipt_sha256=semantic_sha256(changed.semantic_unsigned()),
        )

    candidate = build_massive_adaptive_rl_policy_candidate_v1(
        checkpoint=checkpoint,
        low_cost_trace=trace(10.0, 0.12, frozen=True),
        primary_trace=trace(20.0, 0.08, frozen=False),
        high_cost_trace=trace(40.0, 0.01, frozen=True),
        fixed_control_selection_authority=_fixed_control_authority(tmp_path),
        fixed_control_validation_trace=_fixed_validation_trace(),
    )
    selection_authority = materialize_massive_adaptive_rl_policy_selection_authority_v1(
        root=tmp_path,
        artifact_id="frozen-policy-selection",
        candidates=(candidate,),
        committed_at_ms=10,
    )
    policy = materialize_massive_adaptive_frozen_rl_policy_v1(
        root=tmp_path,
        artifact_id="frozen-policy",
        checkpoint=checkpoint,
        selection_authority=selection_authority,
        committed_at_ms=11,
    )
    assert policy.runtime_policy_replayed
    assert policy.runtime_model_state is not None
    assert not policy.development_outer_policy_authorized

    generic = parse_massive_adaptive_frozen_rl_policy_v1(
        root=tmp_path,
        loaded_source=policy.loaded_source,
    )
    assert generic.runtime_model_state is None
    replayed = authorize_massive_adaptive_frozen_rl_policy_v1(
        root=tmp_path,
        policy=generic,
        checkpoint=checkpoint,
        selection_authority=selection_authority,
    )
    assert replayed.frozen_model_state_receipt_sha256 == (
        policy.frozen_model_state_receipt_sha256
    )
