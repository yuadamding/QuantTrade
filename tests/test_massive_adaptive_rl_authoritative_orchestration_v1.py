from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from rl_quant.evaluation.massive_adaptive_rl_fixed_control_outer_rollout_v1 import (
    authorize_massive_adaptive_rl_fixed_control_outer_rollout_authority_v1,
    materialize_massive_adaptive_rl_fixed_control_outer_rollout_authority_v1,
    parse_massive_adaptive_rl_fixed_control_outer_rollout_authority_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_fixed_control_outer_cost_ladder_v1 import (
    authorize_massive_adaptive_rl_fixed_control_outer_cost_ladder_authority_v1,
    materialize_massive_adaptive_rl_fixed_control_outer_cost_ladder_authority_v1,
    parse_massive_adaptive_rl_fixed_control_outer_cost_ladder_authority_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_rollout_v1 import (
    authorize_massive_adaptive_rl_outer_rollout_authority_v1,
    materialize_massive_adaptive_rl_outer_rollout_authority_v1,
    parse_massive_adaptive_rl_outer_rollout_authority_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_cost_ladder_v1 import (
    authorize_massive_adaptive_rl_outer_cost_ladder_authority_v1,
    materialize_massive_adaptive_rl_outer_cost_ladder_authority_v1,
    parse_massive_adaptive_rl_outer_cost_ladder_authority_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_evidence_v2 import (
    MassiveAdaptiveRLOuterEvidenceV2Error,
    build_massive_adaptive_authenticated_rl_outer_fold_v2,
    build_massive_adaptive_rl_outer_evidence_v2,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_evidence_v3 import (
    MassiveAdaptiveRLOuterEvidenceV3Error,
    build_massive_adaptive_authenticated_rl_outer_fold_v3,
    build_massive_adaptive_rl_outer_evidence_v3,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_evidence_v4 import (
    build_massive_adaptive_authenticated_rl_outer_fold_v4,
    build_massive_adaptive_rl_outer_evidence_v4,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_evidence_authority_v3 import (
    MassiveAdaptiveRLOuterEvidenceAuthorityV3Error,
    authorize_massive_adaptive_rl_outer_evidence_authority_v3,
    materialize_massive_adaptive_rl_outer_evidence_authority_v3,
    parse_massive_adaptive_rl_outer_evidence_authority_v3,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_evidence_authority_v4 import (
    MassiveAdaptiveRLOuterEvidenceAuthorityV4Error,
    authorize_massive_adaptive_rl_outer_evidence_authority_v4,
    materialize_massive_adaptive_rl_outer_evidence_authority_v4,
    parse_massive_adaptive_rl_outer_evidence_authority_v4,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_plan_v2 import (
    MassiveAdaptiveRLOuterPlanV2,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_plan_v3 import (
    build_massive_adaptive_rl_outer_plan_v3,
)
from rl_quant.evaluation.massive_adaptive_rl_policy_evaluator_v1 import (
    evaluate_massive_adaptive_rl_checkpoint_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_cost_ladder_v1 import (
    evaluate_massive_adaptive_rl_checkpoint_cost_ladder_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_cost_ladder_authority_v1 import (
    authorize_massive_adaptive_rl_cost_ladder_authority_v1,
    materialize_massive_adaptive_rl_cost_ladder_authority_v1,
    parse_massive_adaptive_rl_cost_ladder_authority_v1,
)
from rl_quant.evaluation.massive_adaptive_profitability_env_v1 import (
    MassiveAdaptiveProfitabilityEnvV1,
)
from rl_quant.evaluation.massive_adaptive_rl_policy_trace_authority_v1 import (
    authorize_massive_adaptive_rl_policy_trace_authority_v1,
    materialize_massive_adaptive_rl_policy_trace_authority_v1,
    parse_massive_adaptive_rl_policy_trace_authority_v1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.rl.massive_adaptive_ppo_policy_v1 import (
    MassiveAdaptivePPOActorCriticV1,
)
from rl_quant.rl.massive_adaptive_rl_action_v1 import (
    neutral_massive_adaptive_rl_action_v1,
)
from rl_quant.training.massive_adaptive_ppo_v1 import (
    MassiveAdaptivePPOConfigV1,
    MassiveAdaptivePPOTrainerV1,
)
from rl_quant.training.massive_adaptive_prequential_ppo_runner_v1 import (
    MassiveAdaptivePrequentialPPORunnerV1,
    MassiveAdaptivePrequentialPPORunnerV1Error,
)
from rl_quant.training.massive_adaptive_economic_continuity_authority_v1 import (
    build_massive_adaptive_economic_continuity_authority_v1,
)
from rl_quant.training.massive_adaptive_prequential_ppo_checkpoint_authority_v1 import (
    authorize_massive_adaptive_prequential_ppo_checkpoint_authority_v1,
    materialize_massive_adaptive_prequential_ppo_checkpoint_authority_v1,
    parse_massive_adaptive_prequential_ppo_checkpoint_authority_v1,
)
from rl_quant.training.massive_adaptive_rl_chronology_authority_v1 import (
    MASSIVE_ADAPTIVE_RL_CHRONOLOGY_AUTHORITY_V1_SCHEMA,
    MASSIVE_ADAPTIVE_RL_CHRONOLOGY_AUTHORITY_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_CHRONOLOGY_AUTHORITY_V1_SPEC_SHA256,
    MassiveAdaptiveRLChronologyAuthorityV1,
    MassiveAdaptiveRLChronologyAuthorityV1Error,
    bind_massive_adaptive_rl_outer_chronology_v1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_registry_v1 import (
    MassiveAdaptiveRLFixedControlRegistryV1Error,
    build_massive_adaptive_rl_fixed_control_registry_v1,
    registered_massive_adaptive_rl_constant_actions_v1,
    validate_massive_adaptive_rl_fixed_control_registry_coverage_v1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_fit_runner_v1 import (
    authorize_massive_adaptive_rl_fixed_control_fit_authority_v1,
    materialize_massive_adaptive_rl_fixed_control_fit_authority_v1,
    materialize_massive_adaptive_rl_fixed_control_selection_from_fit_v1,
    parse_massive_adaptive_rl_fixed_control_fit_authority_v1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_selection_v1 import (
    build_massive_adaptive_rl_fixed_control_candidate_v1,
    materialize_massive_adaptive_rl_fixed_control_selection_authority_v1,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v1 import (
    build_massive_adaptive_rl_policy_trace_from_identities_v1,
)
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
)
from test_massive_adaptive_profitability_v1_vertical_slice import (
    _adaptive_env_fixture,
)
from test_massive_adaptive_rl_outer_evidence_v1 import _fold


def _digest(value: object) -> str:
    return semantic_sha256(value)


def _outer_plan_v2_for_test(*, cost_fold, fixed_rollout):
    outer_plan_v1 = SimpleNamespace(
        validate=lambda: None,
        fold_index=cost_fold.fold_index,
        semantic_receipt_sha256=cost_fold.outer_plan_receipt_sha256,
        outer_forecast_archive_receipt_sha256=(
            fixed_rollout.policy_trace.forecast_archive_receipt_sha256
        ),
        outer_inference_plan_receipt_sha256=(
            fixed_rollout.policy_trace.inference_plan_receipt_sha256
        ),
        source_data_qualified=False,
    )
    provisional = MassiveAdaptiveRLOuterPlanV2(
        outer_plan_v1=outer_plan_v1,  # type: ignore[arg-type]
        fixed_control_registry_receipt_sha256=_digest("fixed-registry"),
        fixed_control_fit_authority_receipt_sha256=(
            fixed_rollout.fixed_control_fit_authority_receipt_sha256
        ),
        fixed_control_selection_authority_receipt_sha256=(
            fixed_rollout.fixed_control_selection_authority_receipt_sha256
        ),
        selected_fixed_control_id=fixed_rollout.selected_control_id,
        selected_fixed_action_receipt_sha256=(
            fixed_rollout.selected_action_receipt_sha256
        ),
        comparator_frozen_at_ms=1,
        outer_forecast_committed_at_ms=2,
        source_data_qualified=False,
        semantic_receipt_sha256="0" * 64,
        outer_evaluation_authorized=False,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _training_authority(environment):
    dates = tuple(row.decision_session_date for row in environment.inference_plan.rows)
    blocks = tuple(
        SimpleNamespace(
            block_index=index,
            semantic_receipt_sha256=_digest(("block", index)),
            source_forecast_archive_receipt_sha256=(
                environment.forecast_archive.semantic_receipt_sha256
            ),
            calibration_receipt_sha256=environment.calibration.semantic_receipt_sha256,
            forecast_session_dates=(date,),
        )
        for index, date in enumerate(dates)
    )
    receipt = _digest("training-forecast-authority")
    return SimpleNamespace(
        validate=lambda: None,
        blocks=blocks,
        block_inventory_sha256=_digest(
            tuple(block.semantic_receipt_sha256 for block in blocks)
        ),
        origin_session_dates=dates,
        semantic_receipt_sha256=receipt,
        reinforcement_learning_authorized=True,
        source_data_qualified=True,
        outer_fold_index=0,
    )


def _chronology(environment, training_authority):
    dates = tuple(row.decision_session_date for row in environment.inference_plan.rows)
    return SimpleNamespace(
        validate=lambda: None,
        semantic_receipt_sha256=_digest("rl-chronology"),
        fold_index=0,
        training_forecast_authority_receipt_sha256=(
            training_authority.semantic_receipt_sha256
        ),
        rl_fit_origin_dates=dates,
        rl_fit_origin_inventory_sha256=semantic_sha256(dates),
        rl_validation_origin_dates=dates,
        outer_origin_dates=dates,
        validation_inference_plan_receipt_sha256=(
            environment.inference_plan.semantic_receipt_sha256
        ),
        outer_inference_plan_receipt_sha256=(
            environment.inference_plan.semantic_receipt_sha256
        ),
        development_rl_training_authorized=True,
        development_policy_selection_authorized=True,
        outer_evaluation_authorized=True,
        source_data_qualified=True,
    )


def _fixed_selection_fixture(tmp_path, environment, chronology):
    fit_environment = copy.copy(environment)
    fit_environment._state = None
    fit_environment._prepared = None
    fit_environment._observation = None
    fit_environment.reset()
    transitions = []
    neutral = neutral_massive_adaptive_rl_action_v1()
    while True:
        _next, _reward, terminated, truncated, info = fit_environment.step(neutral)
        assert not truncated
        transitions.append(info["transition"])
        if terminated:
            break
    training_trace = build_massive_adaptive_rl_policy_trace_from_identities_v1(
        fold_index=chronology.fold_index,
        checkpoint_receipt_sha256=_digest("fixed-fit-controller"),
        model_state_receipt_sha256=neutral.semantic_receipt_sha256,
        update_index=0,
        training_forecast_authority_receipt_sha256=(
            chronology.training_forecast_authority_receipt_sha256
        ),
        forecast_archive_receipt_sha256=(
            fit_environment.forecast_archive.semantic_receipt_sha256
        ),
        inference_plan_receipt_sha256=(
            fit_environment.inference_plan.semantic_receipt_sha256
        ),
        calibration_receipt_sha256=(
            fit_environment.calibration.semantic_receipt_sha256
        ),
        transaction_cost_basis_points=20.0,
        initial_capital=fit_environment.initial_capital,
        transitions=tuple(transitions),
        frozen_targets_replayed=False,
        evaluation_role="training_control",
        checkpoint_source_data_qualified=False,
    )
    context_receipt = _digest("shared-fixed-fit-context")
    candidates = tuple(
        build_massive_adaptive_rl_fixed_control_candidate_v1(
            fold_index=chronology.fold_index,
            control_id=control_id,
            action=action,
            training_trace=training_trace,
            training_context_receipt_sha256=context_receipt,
        )
        for control_id, action in registered_massive_adaptive_rl_constant_actions_v1()
    )
    selection_authority = (
        materialize_massive_adaptive_rl_fixed_control_selection_authority_v1(
            root=tmp_path,
            artifact_id="outer-fixed-selection",
            candidates=candidates,
            committed_at_ms=7,
        )
    )
    assert selection_authority.runtime_selection is not None
    fit_run = SimpleNamespace(
        fixed_control_registry_receipt_sha256=(
            build_massive_adaptive_rl_fixed_control_registry_v1().semantic_receipt_sha256
        ),
        chronology_authority_receipt_sha256=chronology.semantic_receipt_sha256,
        training_origin_inventory_sha256=(chronology.rl_fit_origin_inventory_sha256),
        training_forecast_authority_receipt_sha256=(
            chronology.training_forecast_authority_receipt_sha256
        ),
        candidate_inventory_sha256=(
            selection_authority.runtime_selection.candidate_inventory_sha256
        ),
        candidates=candidates,
    )
    fit_authority = SimpleNamespace(
        validate=lambda: None,
        runtime_fit_run=fit_run,
        runtime_fit_replayed=True,
        development_control_fit_authorized=False,
        semantic_receipt_sha256=_digest("fixed-fit-authority"),
    )
    return fit_authority, selection_authority


def _checkpoint_authority(environment):
    environment.reset()
    trainer = MassiveAdaptivePPOTrainerV1(
        environment=environment,
        model=MassiveAdaptivePPOActorCriticV1(observation_dim=90),
    )
    checkpoint = trainer.checkpoint()
    forecast_receipt = _digest("training-forecast-authority")
    checkpoint = replace(
        checkpoint,
        training_forecast_authority_receipt_sha256=forecast_receipt,
        semantic_receipt_sha256="0" * 64,
    )
    checkpoint = replace(
        checkpoint,
        semantic_receipt_sha256=semantic_sha256(checkpoint.semantic_unsigned()),
    )
    checkpoint.validate()
    return SimpleNamespace(
        validate=lambda: None,
        semantic_receipt_sha256=_digest("checkpoint-authority"),
        runtime_checkpoint=checkpoint,
        runtime_checkpoint_replayed=True,
        source_data_qualified=False,
    )


def _environment_at_cost(fixture, calibration_values, cost_basis_points: float):
    return MassiveAdaptiveProfitabilityEnvV1(
        forecast_archive=fixture.forecast_archive,
        calibration=calibration_values.calibration,
        inference_plan=fixture.inference_plan,
        decision_roots=fixture.roots,
        context_origins=fixture.contexts,
        fill_source=fixture.fill_source,
        daily_input_authority=fixture.daily,
        identity_authority=fixture.identity,
        economic_event_archive=None,
        initial_capital=10_000_000.0,
        transaction_cost_basis_points=cost_basis_points,
    )


class _ReceiptProxy:
    def __init__(self, target, receipt: str, **overrides) -> None:
        self._target = target
        self.semantic_receipt_sha256 = receipt
        self._overrides = overrides

    def validate(self) -> None:
        self._target.validate()

    def __getattr__(self, name: str):
        try:
            return self._overrides[name]
        except KeyError:
            return getattr(self._target, name)


def _distinct_consecutive_archive_environments():
    _, _, base = _adaptive_env_fixture()
    dates = tuple(row.decision_session_date for row in base.inference_plan.rows)
    assert len(dates) == 2
    environments = []
    for index, date in enumerate(dates):
        environment = copy.copy(base)
        archive_receipt = _digest(("distinct-archive", index))
        calibration_receipt = _digest(("distinct-calibration", index))
        plan_receipt = _digest(("distinct-plan", index))
        row = base.inference_plan.rows[index]
        environment.forecast_archive = _ReceiptProxy(
            base.forecast_archive,
            archive_receipt,
        )
        environment.calibration = _ReceiptProxy(
            base.calibration,
            calibration_receipt,
        )
        environment.inference_plan = _ReceiptProxy(
            base.inference_plan,
            plan_receipt,
            rows=(row,),
        )
        environment.forecasts = {date: base.forecasts[date]}
        environment.roots = {date: base.roots[date]}
        environment.contexts = {date: base.contexts[date]}
        environment.source_inventory_sha256 = _digest(
            (
                archive_receipt,
                calibration_receipt,
                plan_receipt,
                base.source_inventory_sha256,
            )
        )
        environment._state = None
        environment._prepared = None
        environment._observation = None
        environments.append(environment)
    return tuple(environments)


def test_forecast_refit_preserves_book_but_not_position_lock() -> None:
    first, second = _distinct_consecutive_archive_environments()
    dates = (
        first.inference_plan.rows[0].decision_session_date,
        second.inference_plan.rows[0].decision_session_date,
    )
    blocks = tuple(
        SimpleNamespace(
            block_index=index,
            semantic_receipt_sha256=_digest(("distinct-block", index)),
            source_forecast_archive_receipt_sha256=(
                environment.forecast_archive.semantic_receipt_sha256
            ),
            calibration_receipt_sha256=(
                environment.calibration.semantic_receipt_sha256
            ),
            forecast_session_dates=(dates[index],),
        )
        for index, environment in enumerate((first, second))
    )
    authority_receipt = _digest("distinct-training-authority")
    training_authority = SimpleNamespace(
        validate=lambda: None,
        blocks=blocks,
        block_inventory_sha256=_digest(
            tuple(block.semantic_receipt_sha256 for block in blocks)
        ),
        origin_session_dates=dates,
        semantic_receipt_sha256=authority_receipt,
        reinforcement_learning_authorized=True,
        source_data_qualified=True,
        outer_fold_index=0,
    )
    chronology = SimpleNamespace(
        validate=lambda: None,
        semantic_receipt_sha256=_digest("distinct-chronology"),
        fold_index=0,
        training_forecast_authority_receipt_sha256=authority_receipt,
        rl_fit_origin_dates=dates,
        development_rl_training_authorized=True,
    )
    continuity = build_massive_adaptive_economic_continuity_authority_v1(
        previous_block_receipt_sha256=blocks[0].semantic_receipt_sha256,
        next_block_receipt_sha256=blocks[1].semantic_receipt_sha256,
        previous_environment=first,
        next_environment=second,
        source_data_qualified=False,
    )
    assert continuity.carry_books_authorized
    assert not continuity.development_continuity_authorized
    assert (
        continuity.previous_forecast_archive_receipt_sha256
        != continuity.next_forecast_archive_receipt_sha256
    )
    assert (
        continuity.previous_calibration_receipt_sha256
        != continuity.next_calibration_receipt_sha256
    )

    first.reset()
    _, _, terminated, truncated, info = first.step(
        neutral_massive_adaptive_rl_action_v1(),
        continue_economic_episode=True,
    )
    transition = info["transition"]
    assert not terminated
    assert truncated
    assert transition.strategy_terminal_liquidation_cost == 0.0
    assert transition.neutral_terminal_liquidation_cost == 0.0
    assert transition.benchmark_terminal_liquidation_cost == 0.0
    second.restore_continuation(first.state)
    assert second.state.strategy_book == first.state.strategy_book

    runner = MassiveAdaptivePrequentialPPORunnerV1(
        training_authority=training_authority,  # type: ignore[arg-type]
        chronology_authority=chronology,  # type: ignore[arg-type]
        environments={
            first.forecast_archive.semantic_receipt_sha256: first,
            second.forecast_archive.semantic_receipt_sha256: second,
        },
        model=MassiveAdaptivePPOActorCriticV1(observation_dim=90),
        config=MassiveAdaptivePPOConfigV1(
            epochs_per_rollout=1,
            rollout_length=1,
            minibatch_size=1,
        ),
    )
    runner.run_next_update()
    prior_state = first.state
    carried_state = second.state
    assert prior_state.done
    assert carried_state.chronology_cursor == 0
    assert carried_state.strategy_book == prior_state.strategy_book
    assert carried_state.neutral_book == prior_state.neutral_book
    assert carried_state.benchmark_book == prior_state.benchmark_book
    assert carried_state.trailing_state == prior_state.trailing_state
    assert carried_state.previous_action == prior_state.previous_action
    assert carried_state.strategy_book.holdings
    assert runner.checkpoint().continuity_authority_receipts == (
        continuity.semantic_receipt_sha256,
    )

    boundary = runner.checkpoint()
    runner.run_next_update()
    assert runner.training_complete
    uninterrupted = runner.checkpoint()
    assert (
        second.state.strategy_book.high_water_mark
        >= prior_state.strategy_book.high_water_mark
    )

    resumed_first, resumed_second = _distinct_consecutive_archive_environments()
    resumed = MassiveAdaptivePrequentialPPORunnerV1(
        training_authority=training_authority,  # type: ignore[arg-type]
        chronology_authority=chronology,  # type: ignore[arg-type]
        environments={
            resumed_first.forecast_archive.semantic_receipt_sha256: resumed_first,
            resumed_second.forecast_archive.semantic_receipt_sha256: resumed_second,
        },
        model=MassiveAdaptivePPOActorCriticV1(observation_dim=90),
        config=MassiveAdaptivePPOConfigV1(
            epochs_per_rollout=1,
            rollout_length=1,
            minibatch_size=1,
        ),
    )
    resumed.restore(boundary)
    resumed.run_next_update()
    restarted = resumed.checkpoint()
    assert uninterrupted.semantic_receipt_sha256 == restarted.semantic_receipt_sha256
    assert uninterrupted.transition_receipts == restarted.transition_receipts


def test_fixed_controls_are_replayed_over_complete_continuous_fit_tape(
    tmp_path,
) -> None:
    first, second = _distinct_consecutive_archive_environments()
    dates = (
        first.inference_plan.rows[0].decision_session_date,
        second.inference_plan.rows[0].decision_session_date,
    )
    blocks = tuple(
        SimpleNamespace(
            block_index=index,
            semantic_receipt_sha256=_digest(("fixed-fit-block", index)),
            source_forecast_archive_receipt_sha256=(
                environment.forecast_archive.semantic_receipt_sha256
            ),
            calibration_receipt_sha256=(
                environment.calibration.semantic_receipt_sha256
            ),
            forecast_session_dates=(dates[index],),
        )
        for index, environment in enumerate((first, second))
    )
    authority_receipt = _digest("fixed-fit-training-authority")
    training_authority = SimpleNamespace(
        validate=lambda: None,
        blocks=blocks,
        block_inventory_sha256=_digest(
            tuple(block.semantic_receipt_sha256 for block in blocks)
        ),
        origin_session_dates=dates,
        semantic_receipt_sha256=authority_receipt,
        reinforcement_learning_authorized=True,
        source_data_qualified=True,
        outer_fold_index=0,
    )
    chronology = SimpleNamespace(
        validate=lambda: None,
        semantic_receipt_sha256=_digest("fixed-fit-chronology"),
        fold_index=0,
        training_forecast_authority_receipt_sha256=authority_receipt,
        rl_fit_origin_dates=dates,
        rl_fit_origin_inventory_sha256=semantic_sha256(dates),
        source_data_qualified=True,
        development_rl_training_authorized=True,
    )
    environments = {
        first.forecast_archive.semantic_receipt_sha256: first,
        second.forecast_archive.semantic_receipt_sha256: second,
    }
    durable = materialize_massive_adaptive_rl_fixed_control_fit_authority_v1(
        root=tmp_path,
        artifact_id="complete-fixed-grid",
        training_authority=training_authority,  # type: ignore[arg-type]
        chronology_authority=chronology,  # type: ignore[arg-type]
        environments=environments,
        committed_at_ms=20,
    )
    fit = durable.runtime_fit_run
    assert fit is not None
    assert durable.runtime_fit_replayed
    assert not durable.development_control_fit_authorized
    registered_control_count = len(registered_massive_adaptive_rl_constant_actions_v1())
    assert len(fit.traces) == registered_control_count
    assert len(fit.candidates) == registered_control_count
    assert all(row.decision_session_dates == dates for row in fit.traces)
    assert all(row.transition_receipts for row in fit.traces)
    assert len(fit.continuity_authority_receipts) == 1
    selection = materialize_massive_adaptive_rl_fixed_control_selection_from_fit_v1(
        root=tmp_path,
        artifact_id="complete-fixed-grid-selection",
        fit_authority=durable,
        committed_at_ms=21,
    )
    validate_massive_adaptive_rl_fixed_control_registry_coverage_v1(
        registry=build_massive_adaptive_rl_fixed_control_registry_v1(),
        fit_authority=durable,
        selection_authority=selection,
        chronology_authority=chronology,  # type: ignore[arg-type]
    )

    generic = parse_massive_adaptive_rl_fixed_control_fit_authority_v1(
        root=tmp_path,
        loaded_source=durable.loaded_source,
    )
    assert generic.runtime_fit_run is None
    assert not generic.development_control_fit_authorized
    replayed = authorize_massive_adaptive_rl_fixed_control_fit_authority_v1(
        root=tmp_path,
        authority=generic,
        training_authority=training_authority,  # type: ignore[arg-type]
        chronology_authority=chronology,  # type: ignore[arg-type]
        environments=environments,
    )
    assert replayed.semantic_receipt_sha256 == durable.semantic_receipt_sha256


def test_prequential_ppo_uses_every_block_and_resumes_across_boundary(
    tmp_path,
) -> None:
    _, _, environment = _adaptive_env_fixture()
    training_authority = _training_authority(environment)
    chronology = _chronology(environment, training_authority)
    config = MassiveAdaptivePPOConfigV1(
        epochs_per_rollout=1,
        rollout_length=1,
        minibatch_size=1,
    )
    runner = MassiveAdaptivePrequentialPPORunnerV1(
        training_authority=training_authority,  # type: ignore[arg-type]
        chronology_authority=chronology,  # type: ignore[arg-type]
        environments={
            environment.forecast_archive.semantic_receipt_sha256: environment
        },
        model=MassiveAdaptivePPOActorCriticV1(observation_dim=90),
        config=config,
    )
    runner.run_next_update()
    boundary = runner.checkpoint()
    assert boundary.current_block_index == 1
    assert len(boundary.completed_block_receipts) == 1

    durable = materialize_massive_adaptive_prequential_ppo_checkpoint_authority_v1(
        root=tmp_path,
        artifact_id="block-boundary",
        runner=runner,
        committed_at_ms=1,
    )
    assert durable.runtime_checkpoint is not None
    assert (
        durable.runtime_checkpoint.semantic_receipt_sha256
        == boundary.semantic_receipt_sha256
    )
    generic = parse_massive_adaptive_prequential_ppo_checkpoint_authority_v1(
        root=tmp_path,
        loaded_source=durable.loaded_source,
    )
    assert generic.runtime_checkpoint is None

    runner.run_next_update()
    uninterrupted = runner.checkpoint()

    _, _, resumed_environment = _adaptive_env_fixture()
    resumed = MassiveAdaptivePrequentialPPORunnerV1(
        training_authority=training_authority,  # type: ignore[arg-type]
        chronology_authority=chronology,  # type: ignore[arg-type]
        environments={
            resumed_environment.forecast_archive.semantic_receipt_sha256: (
                resumed_environment
            )
        },
        model=MassiveAdaptivePPOActorCriticV1(observation_dim=90),
        config=config,
    )
    reopened = authorize_massive_adaptive_prequential_ppo_checkpoint_authority_v1(
        root=tmp_path,
        authority=generic,
        runner=resumed,
    )
    assert reopened.runtime_checkpoint is not None
    assert (
        reopened.runtime_checkpoint.semantic_receipt_sha256
        == boundary.semantic_receipt_sha256
    )
    resumed.run_next_update()
    restarted = resumed.checkpoint()

    assert uninterrupted.completed_block_receipts == tuple(
        block.semantic_receipt_sha256 for block in training_authority.blocks
    )
    assert uninterrupted.transition_receipts == restarted.transition_receipts
    assert uninterrupted.semantic_receipt_sha256 == restarted.semantic_receipt_sha256
    for name, tensor in uninterrupted.ppo_checkpoint.model_state.items():
        assert torch.equal(tensor, restarted.ppo_checkpoint.model_state[name])


def test_prequential_checkpoint_rejects_outer_and_embedded_provenance_mismatch(
) -> None:
    _, _, environment = _adaptive_env_fixture()
    training_authority = _training_authority(environment)
    chronology = _chronology(environment, training_authority)
    runner = MassiveAdaptivePrequentialPPORunnerV1(
        training_authority=training_authority,  # type: ignore[arg-type]
        chronology_authority=chronology,  # type: ignore[arg-type]
        environments={
            environment.forecast_archive.semantic_receipt_sha256: environment
        },
        model=MassiveAdaptivePPOActorCriticV1(observation_dim=90),
        config=MassiveAdaptivePPOConfigV1(
            epochs_per_rollout=1,
            rollout_length=1,
            minibatch_size=1,
        ),
    )
    runner.run_next_update()
    checkpoint = runner.checkpoint()
    assert checkpoint.transition_receipts
    assert (
        checkpoint.transition_receipts
        == checkpoint.ppo_checkpoint.transition_receipts
    )
    assert (
        checkpoint.transition_source_data_qualified
        == checkpoint.ppo_checkpoint.transition_source_data_qualified
    )
    assert (
        checkpoint.fit_environment_authority_receipts
        == checkpoint.ppo_checkpoint.fit_environment_authority_receipts
    )

    def reseal_outer(value):
        provisional = replace(value, semantic_receipt_sha256="0" * 64)
        return replace(
            provisional,
            semantic_receipt_sha256=semantic_sha256(
                provisional.semantic_unsigned()
            ),
        )

    def reseal_embedded(value):
        provisional = replace(value, semantic_receipt_sha256="0" * 64)
        result = replace(
            provisional,
            semantic_receipt_sha256=semantic_sha256(
                provisional.semantic_unsigned()
            ),
        )
        result.validate()
        return result

    changed_outer_transition = (
        semantic_sha256("changed-outer-transition"),
        *checkpoint.transition_receipts[1:],
    )
    outer_transition_tamper = reseal_outer(
        replace(
            checkpoint,
            transition_receipts=changed_outer_transition,
            transition_inventory_sha256=semantic_sha256(
                changed_outer_transition
            ),
        )
    )
    changed_qualification = (
        not checkpoint.ppo_checkpoint.transition_source_data_qualified[0],
        *checkpoint.ppo_checkpoint.transition_source_data_qualified[1:],
    )
    embedded_qualification_tamper = reseal_outer(
        replace(
            checkpoint,
            ppo_checkpoint=reseal_embedded(
                replace(
                    checkpoint.ppo_checkpoint,
                    transition_source_data_qualified=changed_qualification,
                )
            ),
        )
    )
    outer_environment_tamper = reseal_outer(
        replace(
            checkpoint,
            fit_environment_authority_receipts=(
                semantic_sha256("changed-outer-fit-environment"),
            ),
            fit_environment_authority_inventory_sha256=semantic_sha256(
                (semantic_sha256("changed-outer-fit-environment"),)
            ),
        )
    )
    embedded_environment_tamper = reseal_outer(
        replace(
            checkpoint,
            ppo_checkpoint=reseal_embedded(
                replace(
                    checkpoint.ppo_checkpoint,
                    fit_environment_authority_receipts=(
                        semantic_sha256("changed-embedded-fit-environment"),
                    ),
                )
            ),
        )
    )

    for tampered in (
        outer_transition_tamper,
        embedded_qualification_tamper,
        outer_environment_tamper,
        embedded_environment_tamper,
    ):
        with pytest.raises(
            MassiveAdaptivePrequentialPPORunnerV1Error,
            match="checkpoint differs",
        ):
            tampered.validate()

    training_run = runner.run_to_completion()
    promoted_run = replace(
        training_run,
        source_data_qualified=True,
        development_rl_training_authorized=True,
        semantic_receipt_sha256="0" * 64,
    )
    promoted_run = replace(
        promoted_run,
        semantic_receipt_sha256=semantic_sha256(
            promoted_run.semantic_unsigned()
        ),
    )
    with pytest.raises(
        MassiveAdaptivePrequentialPPORunnerV1Error,
        match="training run differs",
    ):
        promoted_run.validate()


def test_checkpoint_drives_validation_actions_and_trace_replay(tmp_path) -> None:
    _, _, environment = _adaptive_env_fixture()
    training_authority = _training_authority(environment)
    chronology = _chronology(environment, training_authority)
    checkpoint_authority = _checkpoint_authority(environment)

    evaluated = evaluate_massive_adaptive_rl_checkpoint_v1(
        checkpoint_authority=checkpoint_authority,  # type: ignore[arg-type]
        chronology_authority=chronology,  # type: ignore[arg-type]
        environment=environment,
        fold_index=0,
        evaluation_role="inner_validation",
    )
    assert len(evaluated.action_evidence) == len(environment.inference_plan.rows)
    assert tuple(
        row.action_receipt_sha256 for row in evaluated.action_evidence
    ) == tuple(row.action_receipt_sha256 for row in evaluated.transitions)

    authority = materialize_massive_adaptive_rl_policy_trace_authority_v1(
        root=tmp_path,
        artifact_id="checkpoint-policy-trace",
        checkpoint_authority=checkpoint_authority,  # type: ignore[arg-type]
        chronology_authority=chronology,  # type: ignore[arg-type]
        environment=environment,
        fold_index=0,
        evaluation_role="inner_validation",
        committed_at_ms=1,
    )
    assert authority.runtime_trace_replayed
    generic = parse_massive_adaptive_rl_policy_trace_authority_v1(
        root=tmp_path,
        loaded_source=authority.loaded_source,
    )
    assert generic.runtime_trace is None
    replayed = authorize_massive_adaptive_rl_policy_trace_authority_v1(
        root=tmp_path,
        authority=generic,
        checkpoint_authority=checkpoint_authority,  # type: ignore[arg-type]
        chronology_authority=chronology,  # type: ignore[arg-type]
        environment=environment,
    )
    assert replayed.policy_trace_receipt_sha256 == authority.policy_trace_receipt_sha256


def test_checkpoint_cost_ladder_replays_exact_primary_targets(tmp_path) -> None:
    fixture, calibration_values, primary_environment = _adaptive_env_fixture()
    training_authority = _training_authority(primary_environment)
    chronology = _chronology(primary_environment, training_authority)
    checkpoint_authority = _checkpoint_authority(primary_environment)
    ladder = evaluate_massive_adaptive_rl_checkpoint_cost_ladder_v1(
        checkpoint_authority=checkpoint_authority,  # type: ignore[arg-type]
        chronology_authority=chronology,  # type: ignore[arg-type]
        primary_environment=primary_environment,
        low_cost_environment=_environment_at_cost(fixture, calibration_values, 10.0),
        high_cost_environment=_environment_at_cost(fixture, calibration_values, 40.0),
        fold_index=0,
        evaluation_role="inner_validation",
    )

    assert ladder.low_cost_trace.frozen_targets_replayed
    assert ladder.high_cost_trace.frozen_targets_replayed
    assert all(
        row.economic_step.frozen_targets_replayed
        for row in (*ladder.low_cost_transitions, *ladder.high_cost_transitions)
    )
    assert (
        len(
            {
                ladder.low_cost_trace.decision_target_inventory_sha256,
                ladder.primary.policy_trace.decision_target_inventory_sha256,
                ladder.high_cost_trace.decision_target_inventory_sha256,
            }
        )
        == 1
    )

    authority = materialize_massive_adaptive_rl_cost_ladder_authority_v1(
        root=tmp_path,
        artifact_id="checkpoint-cost-ladder",
        checkpoint_authority=checkpoint_authority,  # type: ignore[arg-type]
        chronology_authority=chronology,  # type: ignore[arg-type]
        primary_environment=primary_environment,
        low_cost_environment=_environment_at_cost(fixture, calibration_values, 10.0),
        high_cost_environment=_environment_at_cost(fixture, calibration_values, 40.0),
        fold_index=0,
        evaluation_role="inner_validation",
        committed_at_ms=2,
    )
    assert authority.runtime_ladder_replayed
    generic = parse_massive_adaptive_rl_cost_ladder_authority_v1(
        root=tmp_path,
        loaded_source=authority.loaded_source,
    )
    assert generic.runtime_ladder is None
    reopened = authorize_massive_adaptive_rl_cost_ladder_authority_v1(
        root=tmp_path,
        authority=generic,
        checkpoint_authority=checkpoint_authority,  # type: ignore[arg-type]
        chronology_authority=chronology,  # type: ignore[arg-type]
        primary_environment=primary_environment,
        low_cost_environment=_environment_at_cost(fixture, calibration_values, 10.0),
        high_cost_environment=_environment_at_cost(fixture, calibration_values, 40.0),
    )
    assert reopened.cost_ladder_receipt_sha256 == ladder.semantic_receipt_sha256


def test_frozen_outer_actions_replay_from_attached_policy(tmp_path) -> None:
    fixture, calibration_values, environment = _adaptive_env_fixture()
    training_authority = _training_authority(environment)
    chronology = _chronology(environment, training_authority)
    checkpoint_authority = _checkpoint_authority(environment)
    checkpoint = checkpoint_authority.runtime_checkpoint

    outer_plan_receipt = _digest("outer-plan")
    environment.inference_plan = SimpleNamespace(
        validate=lambda: None,
        fold_index=0,
        rows=environment.inference_plan.rows,
        semantic_receipt_sha256=environment.inference_plan.semantic_receipt_sha256,
        outer_inference_authorized=True,
    )
    frozen_policy = SimpleNamespace(
        validate=lambda: None,
        fold_index=0,
        semantic_receipt_sha256=_digest("frozen-policy"),
        runtime_model_state=checkpoint.model_state,
        runtime_policy_replayed=True,
        selected_rl_checkpoint_receipt_sha256=checkpoint.semantic_receipt_sha256,
        selected_rl_checkpoint_model_state_receipt_sha256=(
            checkpoint.model_state_receipt_sha256
        ),
        selected_update_index=checkpoint.update_index,
        training_forecast_authority_receipt_sha256=(
            checkpoint.training_forecast_authority_receipt_sha256
        ),
        frozen_model_state_receipt_sha256=checkpoint.model_state_receipt_sha256,
        source_data_qualified=False,
        development_outer_policy_authorized=True,
    )
    outer_plan = SimpleNamespace(
        validate=lambda: None,
        fold_index=0,
        semantic_receipt_sha256=outer_plan_receipt,
        frozen_rl_policy_receipt_sha256=frozen_policy.semantic_receipt_sha256,
        frozen_rl_policy_model_state_receipt_sha256=(
            frozen_policy.frozen_model_state_receipt_sha256
        ),
        outer_inference_plan_receipt_sha256=(
            environment.inference_plan.semantic_receipt_sha256
        ),
        outer_forecast_archive_receipt_sha256=(
            environment.forecast_archive.semantic_receipt_sha256
        ),
        calibration_receipt_sha256=environment.calibration.semantic_receipt_sha256,
        compiler_config_receipt_sha256=environment.compiler_config.receipt_sha256,
        primary_capital=environment.initial_capital,
        primary_cost_basis_points=environment.transaction_cost_basis_points,
        outer_evaluation_authorized=True,
    )
    authority = materialize_massive_adaptive_rl_outer_rollout_authority_v1(
        root=tmp_path,
        artifact_id="frozen-outer-rollout",
        outer_plan=outer_plan,  # type: ignore[arg-type]
        frozen_policy=frozen_policy,  # type: ignore[arg-type]
        chronology_authority=chronology,  # type: ignore[arg-type]
        environment=environment,
        committed_at_ms=2,
    )
    assert authority.runtime_rollout_replayed
    assert authority.runtime_rollout is not None
    assert tuple(
        row.action_receipt_sha256 for row in authority.runtime_rollout.action_evidence
    ) == tuple(
        row.action_receipt_sha256 for row in authority.runtime_rollout.transitions
    )

    generic = parse_massive_adaptive_rl_outer_rollout_authority_v1(
        root=tmp_path,
        loaded_source=authority.loaded_source,
    )
    replayed = authorize_massive_adaptive_rl_outer_rollout_authority_v1(
        root=tmp_path,
        authority=generic,
        outer_plan=outer_plan,  # type: ignore[arg-type]
        frozen_policy=frozen_policy,  # type: ignore[arg-type]
        chronology_authority=chronology,  # type: ignore[arg-type]
        environment=environment,
    )
    assert (
        replayed.outer_rollout_receipt_sha256 == authority.outer_rollout_receipt_sha256
    )

    registry = build_massive_adaptive_rl_fixed_control_registry_v1()
    fit_authority, selection_authority = _fixed_selection_fixture(
        tmp_path,
        environment,
        chronology,
    )
    fixed_environment = _environment_at_cost(fixture, calibration_values, 20.0)
    fixed_environment.inference_plan = environment.inference_plan
    fixed_authority = (
        materialize_massive_adaptive_rl_fixed_control_outer_rollout_authority_v1(
            root=tmp_path,
            artifact_id="fit-selected-fixed-outer-rollout",
            outer_plan=outer_plan,  # type: ignore[arg-type]
            registry=registry,
            fit_authority=fit_authority,  # type: ignore[arg-type]
            selection_authority=selection_authority,
            chronology_authority=chronology,  # type: ignore[arg-type]
            environment=fixed_environment,
            committed_at_ms=8,
        )
    )
    assert fixed_authority.runtime_rollout_replayed
    assert fixed_authority.runtime_rollout is not None
    assert fixed_authority.runtime_rollout.environment_source_inventory_sha256 == (
        authority.runtime_rollout.environment_source_inventory_sha256
    )
    generic_fixed = parse_massive_adaptive_rl_fixed_control_outer_rollout_authority_v1(
        root=tmp_path,
        loaded_source=fixed_authority.loaded_source,
    )
    reopened_fixed = (
        authorize_massive_adaptive_rl_fixed_control_outer_rollout_authority_v1(
            root=tmp_path,
            authority=generic_fixed,
            outer_plan=outer_plan,  # type: ignore[arg-type]
            registry=registry,
            fit_authority=fit_authority,  # type: ignore[arg-type]
            selection_authority=selection_authority,
            chronology_authority=chronology,  # type: ignore[arg-type]
            environment=fixed_environment,
        )
    )
    assert (
        reopened_fixed.outer_rollout_receipt_sha256
        == fixed_authority.outer_rollout_receipt_sha256
    )

    fixed_low_environment = _environment_at_cost(fixture, calibration_values, 10.0)
    fixed_high_environment = _environment_at_cost(fixture, calibration_values, 40.0)
    fixed_low_environment.inference_plan = environment.inference_plan
    fixed_high_environment.inference_plan = environment.inference_plan
    fixed_cost_authority = (
        materialize_massive_adaptive_rl_fixed_control_outer_cost_ladder_authority_v1(
            root=tmp_path,
            artifact_id="fit-selected-fixed-outer-cost-ladder",
            rollout_authority=fixed_authority,
            primary_environment=fixed_environment,
            low_cost_environment=fixed_low_environment,
            high_cost_environment=fixed_high_environment,
            committed_at_ms=9,
        )
    )
    assert fixed_cost_authority.runtime_ladder_replayed
    assert fixed_cost_authority.runtime_ladder is not None
    assert fixed_cost_authority.runtime_ladder.low_cost_trace.frozen_targets_replayed
    assert (
        fixed_cost_authority.runtime_ladder.low_cost_trace.decision_target_inventory_sha256
        == fixed_cost_authority.runtime_ladder.primary_trace.decision_target_inventory_sha256
        == fixed_cost_authority.runtime_ladder.high_cost_trace.decision_target_inventory_sha256
    )
    generic_fixed_cost = (
        parse_massive_adaptive_rl_fixed_control_outer_cost_ladder_authority_v1(
            root=tmp_path,
            loaded_source=fixed_cost_authority.loaded_source,
        )
    )
    replayed_fixed_cost = (
        authorize_massive_adaptive_rl_fixed_control_outer_cost_ladder_authority_v1(
            root=tmp_path,
            authority=generic_fixed_cost,
            rollout_authority=fixed_authority,
            primary_environment=fixed_environment,
            low_cost_environment=_environment_at_cost(
                fixture, calibration_values, 10.0
            ),
            high_cost_environment=_environment_at_cost(
                fixture, calibration_values, 40.0
            ),
        )
    )
    assert (
        replayed_fixed_cost.fixed_control_outer_cost_ladder_receipt_sha256
        == fixed_cost_authority.fixed_control_outer_cost_ladder_receipt_sha256
    )

    low_environment = _environment_at_cost(fixture, calibration_values, 10.0)
    high_environment = _environment_at_cost(fixture, calibration_values, 40.0)
    low_environment.inference_plan = environment.inference_plan
    high_environment.inference_plan = environment.inference_plan
    cost_authority = materialize_massive_adaptive_rl_outer_cost_ladder_authority_v1(
        root=tmp_path,
        artifact_id="frozen-outer-cost-ladder",
        rollout_authority=authority,
        primary_environment=environment,
        low_cost_environment=low_environment,
        high_cost_environment=high_environment,
        committed_at_ms=3,
    )
    assert cost_authority.runtime_ladder_replayed
    assert cost_authority.runtime_ladder is not None
    assert cost_authority.runtime_ladder.low_cost_trace.frozen_targets_replayed
    generic_cost = parse_massive_adaptive_rl_outer_cost_ladder_authority_v1(
        root=tmp_path,
        loaded_source=cost_authority.loaded_source,
    )
    reopened_cost = authorize_massive_adaptive_rl_outer_cost_ladder_authority_v1(
        root=tmp_path,
        authority=generic_cost,
        rollout_authority=authority,
        primary_environment=environment,
        low_cost_environment=_environment_at_cost(fixture, calibration_values, 10.0),
        high_cost_environment=_environment_at_cost(fixture, calibration_values, 40.0),
    )
    assert (
        reopened_cost.outer_cost_ladder_receipt_sha256
        == cost_authority.outer_cost_ladder_receipt_sha256
    )


def test_rl_chronology_and_fixed_control_registry_fail_closed() -> None:
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_CHRONOLOGY_AUTHORITY_V1_SCHEMA,
        "fold_index": 0,
        "training_forecast_authority_receipt_sha256": _digest("train"),
        "validation_inference_plan_receipt_sha256": _digest("validation"),
        "split_plan_receipt_sha256": _digest("split"),
        "outer_fold_receipt_sha256": _digest("fold"),
        "outer_inference_plan_receipt_sha256": _digest("outer"),
        "rl_fit_origin_dates": ("2020-01-02",),
        "rl_validation_origin_dates": ("2020-02-03",),
        "outer_origin_dates": ("2020-03-02",),
        "rl_fit_origin_inventory_sha256": semantic_sha256(("2020-01-02",)),
        "rl_validation_origin_inventory_sha256": semantic_sha256(("2020-02-03",)),
        "outer_origin_inventory_sha256": semantic_sha256(("2020-03-02",)),
        "source_data_qualified": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_CHRONOLOGY_AUTHORITY_V1_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_CHRONOLOGY_AUTHORITY_V1_SOURCE_SHA256
        ),
    }
    chronology = MassiveAdaptiveRLChronologyAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        development_rl_training_authorized=True,
        development_policy_selection_authorized=True,
        outer_evaluation_authorized=True,
    )
    chronology.validate()
    unbound = replace(
        chronology,
        outer_inference_plan_receipt_sha256=None,
        semantic_receipt_sha256="0" * 64,
        outer_evaluation_authorized=False,
    )
    unbound = replace(
        unbound,
        semantic_receipt_sha256=semantic_sha256(unbound.semantic_unsigned()),
    )
    unbound.validate()
    rebound = bind_massive_adaptive_rl_outer_chronology_v1(
        chronology_authority=unbound,
        outer_inference_plan=SimpleNamespace(
            validate=lambda: None,
            outer_inference_authorized=True,
            fold_index=0,
            split_plan_receipt_sha256=unbound.split_plan_receipt_sha256,
            rows=tuple(
                SimpleNamespace(decision_session_date=value)
                for value in unbound.outer_origin_dates
            ),
            semantic_receipt_sha256=_digest("reopened-outer-plan"),
        ),  # type: ignore[arg-type]
    )
    assert rebound.outer_evaluation_authorized
    assert rebound.outer_inference_plan_receipt_sha256 == _digest("reopened-outer-plan")
    overlap = replace(
        chronology,
        rl_validation_origin_dates=chronology.rl_fit_origin_dates,
        rl_validation_origin_inventory_sha256=chronology.rl_fit_origin_inventory_sha256,
        semantic_receipt_sha256="0" * 64,
    )
    overlap = replace(
        overlap,
        semantic_receipt_sha256=semantic_sha256(overlap.semantic_unsigned()),
    )
    with pytest.raises(MassiveAdaptiveRLChronologyAuthorityV1Error):
        overlap.validate()

    registry = build_massive_adaptive_rl_fixed_control_registry_v1()
    assert registry.control_ids == (
        "FC00",
        "FC01",
        "FC02",
        "FC03",
        "FC04",
        "FC05",
        "FC07",
        "FC08",
        "FC09",
        "FC10",
        "FC11",
        "FC12",
        "FC06",
    )
    actions = dict(registered_massive_adaptive_rl_constant_actions_v1())
    assert actions["FC03"].uncertainty_control == -actions["FC07"].uncertainty_control
    assert actions["FC04"].risk_control == -actions["FC08"].risk_control
    assert actions["FC05"].trade_cost_control == -actions["FC09"].trade_cost_control
    incomplete = SimpleNamespace(
        validate=lambda: None,
        runtime_selection_replayed=True,
        runtime_candidates=(),
    )
    incomplete_fit = SimpleNamespace(
        validate=lambda: None,
        runtime_fit_run=None,
        runtime_fit_replayed=False,
    )
    with pytest.raises(
        MassiveAdaptiveRLFixedControlRegistryV1Error,
        match="incomplete",
    ):
        validate_massive_adaptive_rl_fixed_control_registry_coverage_v1(
            registry=registry,
            fit_authority=incomplete_fit,  # type: ignore[arg-type]
            selection_authority=incomplete,  # type: ignore[arg-type]
            chronology_authority=chronology,
        )


def test_outer_evidence_v2_requires_frozen_policy_rollout_authority(tmp_path) -> None:
    authenticated = []
    rollout_authorities = []
    fixed_authorities = []
    outer_plans_v2 = []
    for index in range(4):
        cost_fold = _fold(index)
        decision_dates = tuple(
            f"2024-{index + 1:02d}-{day + 1:02d}" for day in range(126)
        )
        forecast_receipt = _digest(("outer-forecast", index))
        plan_receipt = _digest(("outer-inference", index))
        calibration_receipt = _digest(("outer-calibration", index))
        environment_receipt = _digest(("outer-environment", index))
        rollout = SimpleNamespace(
            validate=lambda: None,
            fold_index=index,
            frozen_policy_receipt_sha256=(cost_fold.frozen_rl_policy_receipt_sha256),
            policy_trace=SimpleNamespace(
                semantic_receipt_sha256=cost_fold.primary_trace_receipt_sha256,
                decision_session_dates=decision_dates,
                forecast_archive_receipt_sha256=forecast_receipt,
                inference_plan_receipt_sha256=plan_receipt,
                calibration_receipt_sha256=calibration_receipt,
                initial_capital=10_000_000.0,
            ),
            decision_target_inventory_sha256=(
                cost_fold.decision_target_inventory_sha256
            ),
            environment_source_inventory_sha256=environment_receipt,
            semantic_receipt_sha256=_digest(("rollout", index)),
        )
        authority = SimpleNamespace(
            validate=lambda: None,
            runtime_rollout=rollout,
            runtime_rollout_replayed=True,
            outer_evaluation_authorized=False,
            semantic_receipt_sha256=_digest(("rollout-authority", index)),
        )
        rollout_authorities.append(authority)
        cost_ladder = SimpleNamespace(
            outer_rollout_authority_receipt_sha256=(authority.semantic_receipt_sha256),
            outer_rollout_receipt_sha256=rollout.semantic_receipt_sha256,
            frozen_policy_receipt_sha256=(cost_fold.frozen_rl_policy_receipt_sha256),
            primary_trace=rollout.policy_trace,
            low_cost_trace=SimpleNamespace(
                semantic_receipt_sha256=cost_fold.low_cost_trace_receipt_sha256
            ),
            high_cost_trace=SimpleNamespace(
                semantic_receipt_sha256=cost_fold.high_cost_trace_receipt_sha256
            ),
            decision_target_inventory_sha256=(
                cost_fold.decision_target_inventory_sha256
            ),
            semantic_receipt_sha256=_digest(("cost-ladder", index)),
        )
        cost_authority = SimpleNamespace(
            validate=lambda: None,
            runtime_ladder=cost_ladder,
            runtime_ladder_replayed=True,
            outer_evaluation_authorized=False,
            semantic_receipt_sha256=_digest(("cost-authority", index)),
        )
        authenticated_fold = build_massive_adaptive_authenticated_rl_outer_fold_v2(
            cost_fold=cost_fold,
            rollout_authority=authority,  # type: ignore[arg-type]
            cost_ladder_authority=cost_authority,  # type: ignore[arg-type]
        )
        authenticated.append(authenticated_fold)
        fixed_rollout = SimpleNamespace(
            fold_index=index,
            outer_plan_receipt_sha256=cost_fold.outer_plan_receipt_sha256,
            fixed_control_fit_authority_receipt_sha256=_digest(("fit", index)),
            fixed_control_selection_authority_receipt_sha256=_digest(
                ("fixed-selection", index)
            ),
            selected_control_id="FC06",
            selected_action_receipt_sha256=_digest(("fixed-action", index)),
            policy_trace=SimpleNamespace(
                semantic_receipt_sha256=(
                    cost_fold.best_fixed_control_trace_receipt_sha256
                ),
                evaluation_role="outer_test",
                transaction_cost_basis_points=20.0,
                frozen_targets_replayed=False,
                decision_session_dates=decision_dates,
                forecast_archive_receipt_sha256=forecast_receipt,
                inference_plan_receipt_sha256=plan_receipt,
                calibration_receipt_sha256=calibration_receipt,
                initial_capital=10_000_000.0,
            ),
            environment_source_inventory_sha256=environment_receipt,
            semantic_receipt_sha256=_digest(("fixed-rollout", index)),
        )
        fixed_authorities.append(
            SimpleNamespace(
                validate=lambda: None,
                runtime_rollout=fixed_rollout,
                runtime_rollout_replayed=True,
                outer_evaluation_authorized=False,
                environment_source_inventory_sha256=environment_receipt,
                semantic_receipt_sha256=_digest(("fixed-authority", index)),
            )
        )
        outer_plans_v2.append(
            _outer_plan_v2_for_test(
                cost_fold=cost_fold,
                fixed_rollout=fixed_rollout,
            )
        )
    evidence = build_massive_adaptive_rl_outer_evidence_v2(authenticated)
    assert evidence.evidence_v1.incremental_rl_log_return_lcb95 > 0.0
    assert not evidence.source_data_qualified

    authenticated_v3 = tuple(
        build_massive_adaptive_authenticated_rl_outer_fold_v3(
            authenticated_fold_v2=fold,
            outer_plan_v2=outer_plan_v2,
            ppo_outer_rollout_authority=rollout_authority,  # type: ignore[arg-type]
            fixed_control_outer_authority=fixed_authority,  # type: ignore[arg-type]
        )
        for fold, outer_plan_v2, rollout_authority, fixed_authority in zip(
            authenticated,
            outer_plans_v2,
            rollout_authorities,
            fixed_authorities,
            strict=True,
        )
    )
    evidence_v3 = build_massive_adaptive_rl_outer_evidence_v3(authenticated_v3)
    assert evidence_v3.evidence_v2.semantic_receipt_sha256 == (
        evidence.semantic_receipt_sha256
    )
    assert not evidence_v3.profitability_reporting_authorized

    authenticated_v4 = []
    outer_access_commitments = []
    gated_outer_forecast_archives = []
    outer_plans_v3 = []
    ppo_cost_authorities = []
    fixed_cost_authorities = []
    for index, (fold_v3, plan_v2, fixed_authority) in enumerate(
        zip(
            authenticated_v3,
            outer_plans_v2,
            fixed_authorities,
            strict=True,
        )
    ):
        cost_fold = fold_v3.authenticated_fold_v2.cost_fold
        dates = fixed_authority.runtime_rollout.policy_trace.decision_session_dates
        forecast_receipt = (
            fixed_authority.runtime_rollout.policy_trace.forecast_archive_receipt_sha256
        )
        inference_receipt = (
            fixed_authority.runtime_rollout.policy_trace.inference_plan_receipt_sha256
        )
        calibration_receipt = (
            fixed_authority.runtime_rollout.policy_trace.calibration_receipt_sha256
        )
        ppo_high = SimpleNamespace(
            semantic_receipt_sha256=cost_fold.high_cost_trace_receipt_sha256,
            decision_session_dates=dates,
            forecast_archive_receipt_sha256=forecast_receipt,
            inference_plan_receipt_sha256=inference_receipt,
            calibration_receipt_sha256=calibration_receipt,
            initial_capital=10_000_000.0,
            incremental_rl_log_returns=(0.002,) * len(dates),
        )
        ppo_ladder = SimpleNamespace(
            high_cost_trace=ppo_high,
            semantic_receipt_sha256=(
                fold_v3.authenticated_fold_v2.outer_cost_ladder_receipt_sha256
            ),
        )
        ppo_cost_authority = SimpleNamespace(
            validate=lambda: None,
            runtime_ladder=ppo_ladder,
            runtime_ladder_replayed=True,
            outer_evaluation_authorized=False,
            semantic_receipt_sha256=(
                fold_v3.authenticated_fold_v2.outer_cost_ladder_authority_receipt_sha256
            ),
        )
        fixed_high = SimpleNamespace(
            semantic_receipt_sha256=_digest(("fixed-high-trace", index)),
            decision_session_dates=dates,
            forecast_archive_receipt_sha256=forecast_receipt,
            inference_plan_receipt_sha256=inference_receipt,
            calibration_receipt_sha256=calibration_receipt,
            initial_capital=10_000_000.0,
            incremental_rl_log_returns=(0.001,) * len(dates),
        )
        fixed_ladder = SimpleNamespace(
            fixed_control_outer_rollout_authority_receipt_sha256=(
                fold_v3.fixed_control_outer_authority_receipt_sha256
            ),
            fixed_control_outer_rollout_receipt_sha256=(
                fold_v3.fixed_control_outer_rollout_receipt_sha256
            ),
            primary_trace=fixed_authority.runtime_rollout.policy_trace,
            high_cost_trace=fixed_high,
            selected_control_id=plan_v2.selected_fixed_control_id,
            selected_action_receipt_sha256=(
                plan_v2.selected_fixed_action_receipt_sha256
            ),
            semantic_receipt_sha256=_digest(("fixed-cost-ladder", index)),
        )
        fixed_cost_authority = SimpleNamespace(
            validate=lambda: None,
            runtime_ladder=fixed_ladder,
            runtime_ladder_replayed=True,
            outer_evaluation_authorized=False,
            semantic_receipt_sha256=_digest(("fixed-cost-authority", index)),
        )
        outer_access_commitment = SimpleNamespace(
            validate=lambda: None,
            fold_index=index,
            outer_forecast_access_authorized=True,
            outer_inference_plan_receipt_sha256=(
                plan_v2.outer_plan_v1.outer_inference_plan_receipt_sha256
            ),
            fixed_control_fit_authority_receipt_sha256=(
                plan_v2.fixed_control_fit_authority_receipt_sha256
            ),
            fixed_control_selection_authority_receipt_sha256=(
                plan_v2.fixed_control_selection_authority_receipt_sha256
            ),
            selected_fixed_action_receipt_sha256=(
                plan_v2.selected_fixed_action_receipt_sha256
            ),
            semantic_receipt_sha256=_digest(("outer-access-commitment", index)),
        )
        gated_outer_forecast_archive = SimpleNamespace(
            validate=lambda: None,
            fold_index=index,
            outer_forecast_authorized=True,
            outer_access_commitment_receipt_sha256=(
                outer_access_commitment.semantic_receipt_sha256
            ),
            raw_outer_forecast_archive_receipt_sha256=(
                plan_v2.outer_plan_v1.outer_forecast_archive_receipt_sha256
            ),
            semantic_receipt_sha256=_digest(("gated-outer-forecast", index)),
        )
        plan_v3 = build_massive_adaptive_rl_outer_plan_v3(
            outer_plan_v2=plan_v2,
            outer_access_commitment=outer_access_commitment,  # type: ignore[arg-type]
            gated_outer_forecast_archive=(
                gated_outer_forecast_archive  # type: ignore[arg-type]
            ),
        )
        assert not plan_v3.source_data_qualified
        assert not plan_v3.outer_evaluation_authorized
        outer_access_commitments.append(outer_access_commitment)
        gated_outer_forecast_archives.append(gated_outer_forecast_archive)
        outer_plans_v3.append(plan_v3)
        ppo_cost_authorities.append(ppo_cost_authority)
        fixed_cost_authorities.append(fixed_cost_authority)
        authenticated_v4.append(
            build_massive_adaptive_authenticated_rl_outer_fold_v4(
                authenticated_fold_v3=fold_v3,
                outer_plan_v3=plan_v3,  # type: ignore[arg-type]
                ppo_cost_ladder_authority=ppo_cost_authority,  # type: ignore[arg-type]
                fixed_control_cost_ladder_authority=(
                    fixed_cost_authority  # type: ignore[arg-type]
                ),
            )
        )
    evidence_v4 = build_massive_adaptive_rl_outer_evidence_v4(authenticated_v4)
    assert evidence_v4.high_cost_ppo_minus_fixed_control_nonnegative
    assert (
        evidence_v4.mean_high_cost_ppo_minus_fixed_control_log_return
        == pytest.approx(0.001)
    )
    assert not evidence_v4.profitability_reporting_authorized

    durable = materialize_massive_adaptive_rl_outer_evidence_authority_v3(
        root=tmp_path,
        artifact_id="authenticated-v3-evidence",
        outer_plans_v2=outer_plans_v2,
        authenticated_folds_v2=authenticated,
        ppo_outer_rollout_authorities=rollout_authorities,  # type: ignore[arg-type]
        fixed_control_outer_authorities=fixed_authorities,  # type: ignore[arg-type]
        committed_at_ms=10,
    )
    assert durable.runtime_evidence_replayed
    assert durable.runtime_evidence is not None
    assert durable.runtime_evidence.semantic_receipt_sha256 == (
        evidence_v3.semantic_receipt_sha256
    )
    durable_v4 = materialize_massive_adaptive_rl_outer_evidence_authority_v4(
        root=tmp_path,
        artifact_id="authenticated-v4-evidence",
        evidence_v3_authority=durable,
        outer_plans_v2=outer_plans_v2,
        outer_access_commitments=(
            outer_access_commitments  # type: ignore[arg-type]
        ),
        gated_outer_forecast_archives=(
            gated_outer_forecast_archives  # type: ignore[arg-type]
        ),
        outer_plans_v3=outer_plans_v3,  # type: ignore[arg-type]
        ppo_cost_ladder_authorities=ppo_cost_authorities,  # type: ignore[arg-type]
        fixed_control_cost_ladder_authorities=(
            fixed_cost_authorities  # type: ignore[arg-type]
        ),
        committed_at_ms=11,
    )
    assert durable_v4.runtime_evidence_replayed
    assert durable_v4.runtime_evidence is not None
    assert durable_v4.runtime_evidence.semantic_receipt_sha256 == (
        evidence_v4.semantic_receipt_sha256
    )
    generic_v4 = parse_massive_adaptive_rl_outer_evidence_authority_v4(
        root=tmp_path,
        loaded_source=durable_v4.loaded_source,
    )
    assert not generic_v4.runtime_evidence_replayed
    invented_plan_v3 = replace(
        outer_plans_v3[0],
        outer_access_commitment_receipt_sha256=_digest(
            "invented-outer-access-commitment"
        ),
        gated_outer_forecast_archive_receipt_sha256=_digest(
            "invented-gated-outer-forecast"
        ),
        semantic_receipt_sha256="0" * 64,
    )
    invented_plan_v3 = replace(
        invented_plan_v3,
        semantic_receipt_sha256=semantic_sha256(invented_plan_v3.semantic_unsigned()),
    )
    invented_plan_v3.validate()
    with pytest.raises(
        MassiveAdaptiveRLOuterEvidenceAuthorityV4Error,
        match="differs from the committed access path",
    ):
        authorize_massive_adaptive_rl_outer_evidence_authority_v4(
            root=tmp_path,
            authority=generic_v4,
            evidence_v3_authority=durable,
            outer_plans_v2=outer_plans_v2,
            outer_access_commitments=(
                outer_access_commitments  # type: ignore[arg-type]
            ),
            gated_outer_forecast_archives=(
                gated_outer_forecast_archives  # type: ignore[arg-type]
            ),
            outer_plans_v3=(invented_plan_v3, *outer_plans_v3[1:]),
            ppo_cost_ladder_authorities=(
                ppo_cost_authorities  # type: ignore[arg-type]
            ),
            fixed_control_cost_ladder_authorities=(
                fixed_cost_authorities  # type: ignore[arg-type]
            ),
        )
    reopened_v4 = authorize_massive_adaptive_rl_outer_evidence_authority_v4(
        root=tmp_path,
        authority=generic_v4,
        evidence_v3_authority=durable,
        outer_plans_v2=outer_plans_v2,
        outer_access_commitments=(
            outer_access_commitments  # type: ignore[arg-type]
        ),
        gated_outer_forecast_archives=(
            gated_outer_forecast_archives  # type: ignore[arg-type]
        ),
        outer_plans_v3=outer_plans_v3,  # type: ignore[arg-type]
        ppo_cost_ladder_authorities=ppo_cost_authorities,  # type: ignore[arg-type]
        fixed_control_cost_ladder_authorities=(
            fixed_cost_authorities  # type: ignore[arg-type]
        ),
    )
    assert reopened_v4.semantic_receipt_sha256 == durable_v4.semantic_receipt_sha256
    generic = parse_massive_adaptive_rl_outer_evidence_authority_v3(
        root=tmp_path,
        loaded_source=durable.loaded_source,
    )
    assert generic.runtime_evidence is None
    assert not generic.outer_development_conclusion_authorized

    substituted = replace(
        outer_plans_v2[0],
        selected_fixed_action_receipt_sha256=_digest("substituted-fixed-action"),
        semantic_receipt_sha256="0" * 64,
    )
    substituted = replace(
        substituted,
        semantic_receipt_sha256=semantic_sha256(substituted.semantic_unsigned()),
    )
    substituted.validate()
    with pytest.raises(MassiveAdaptiveRLOuterEvidenceAuthorityV3Error):
        authorize_massive_adaptive_rl_outer_evidence_authority_v3(
            root=tmp_path,
            authority=generic,
            outer_plans_v2=(substituted, *outer_plans_v2[1:]),
            authenticated_folds_v2=authenticated,
            ppo_outer_rollout_authorities=rollout_authorities,  # type: ignore[arg-type]
            fixed_control_outer_authorities=fixed_authorities,  # type: ignore[arg-type]
        )

    mismatched_rollout_values = vars(fixed_authorities[0].runtime_rollout).copy()
    mismatched_rollout_values["environment_source_inventory_sha256"] = _digest(
        "wrong-outer-environment"
    )
    mismatched_fixed = SimpleNamespace(
        validate=lambda: None,
        runtime_rollout=SimpleNamespace(**mismatched_rollout_values),
        runtime_rollout_replayed=True,
        outer_evaluation_authorized=False,
        environment_source_inventory_sha256=_digest("wrong-outer-environment"),
    )
    with pytest.raises(MassiveAdaptiveRLOuterEvidenceV3Error, match="market context"):
        build_massive_adaptive_authenticated_rl_outer_fold_v3(
            authenticated_fold_v2=authenticated[0],
            outer_plan_v2=outer_plans_v2[0],
            ppo_outer_rollout_authority=rollout_authorities[0],  # type: ignore[arg-type]
            fixed_control_outer_authority=mismatched_fixed,  # type: ignore[arg-type]
        )

    changed_rollout = replace(
        authenticated[0].cost_fold,
        primary_trace_receipt_sha256=_digest("unattached-trace"),
        semantic_receipt_sha256="0" * 64,
    )
    changed_rollout = replace(
        changed_rollout,
        semantic_receipt_sha256=semantic_sha256(changed_rollout.semantic_unsigned()),
    )
    with pytest.raises(MassiveAdaptiveRLOuterEvidenceV2Error, match="not derived"):
        build_massive_adaptive_authenticated_rl_outer_fold_v2(
            cost_fold=changed_rollout,
            rollout_authority=SimpleNamespace(
                validate=lambda: None,
                runtime_rollout=SimpleNamespace(
                    fold_index=0,
                    frozen_policy_receipt_sha256=(
                        changed_rollout.frozen_rl_policy_receipt_sha256
                    ),
                    policy_trace=SimpleNamespace(
                        semantic_receipt_sha256=_fold(0).primary_trace_receipt_sha256
                    ),
                    decision_target_inventory_sha256=(
                        changed_rollout.decision_target_inventory_sha256
                    ),
                ),
                runtime_rollout_replayed=True,
            ),  # type: ignore[arg-type]
            cost_ladder_authority=SimpleNamespace(
                validate=lambda: None,
                runtime_ladder=None,
                runtime_ladder_replayed=False,
            ),  # type: ignore[arg-type]
        )
