from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

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
from rl_quant.training.massive_adaptive_ppo_v1 import (
    MassiveAdaptivePPOConfigV1,
    MassiveAdaptivePPOTrainerV1,
)
from rl_quant.training.massive_adaptive_prequential_ppo_runner_v1 import (
    MassiveAdaptivePrequentialPPORunnerV1,
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
    validate_massive_adaptive_rl_fixed_control_registry_coverage_v1,
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
    )


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
        development_rl_training_authorized=True,
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
    assert tuple(row.action_receipt_sha256 for row in evaluated.action_evidence) == tuple(
        row.action_receipt_sha256 for row in evaluated.transitions
    )

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
        low_cost_environment=_environment_at_cost(
            fixture, calibration_values, 10.0
        ),
        high_cost_environment=_environment_at_cost(
            fixture, calibration_values, 40.0
        ),
        fold_index=0,
        evaluation_role="inner_validation",
    )

    assert ladder.low_cost_trace.frozen_targets_replayed
    assert ladder.high_cost_trace.frozen_targets_replayed
    assert all(
        row.economic_step.frozen_targets_replayed
        for row in (*ladder.low_cost_transitions, *ladder.high_cost_transitions)
    )
    assert len(
        {
            ladder.low_cost_trace.decision_target_inventory_sha256,
            ladder.primary.policy_trace.decision_target_inventory_sha256,
            ladder.high_cost_trace.decision_target_inventory_sha256,
        }
    ) == 1

    authority = materialize_massive_adaptive_rl_cost_ladder_authority_v1(
        root=tmp_path,
        artifact_id="checkpoint-cost-ladder",
        checkpoint_authority=checkpoint_authority,  # type: ignore[arg-type]
        chronology_authority=chronology,  # type: ignore[arg-type]
        primary_environment=primary_environment,
        low_cost_environment=_environment_at_cost(
            fixture, calibration_values, 10.0
        ),
        high_cost_environment=_environment_at_cost(
            fixture, calibration_values, 40.0
        ),
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
        low_cost_environment=_environment_at_cost(
            fixture, calibration_values, 10.0
        ),
        high_cost_environment=_environment_at_cost(
            fixture, calibration_values, 40.0
        ),
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
        row.action_receipt_sha256
        for row in authority.runtime_rollout.action_evidence
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
    assert replayed.outer_rollout_receipt_sha256 == authority.outer_rollout_receipt_sha256

    low_environment = _environment_at_cost(fixture, calibration_values, 10.0)
    high_environment = _environment_at_cost(fixture, calibration_values, 40.0)
    low_environment.inference_plan = environment.inference_plan
    high_environment.inference_plan = environment.inference_plan
    cost_authority = (
        materialize_massive_adaptive_rl_outer_cost_ladder_authority_v1(
            root=tmp_path,
            artifact_id="frozen-outer-cost-ladder",
            rollout_authority=authority,
            primary_environment=environment,
            low_cost_environment=low_environment,
            high_cost_environment=high_environment,
            committed_at_ms=3,
        )
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
        low_cost_environment=_environment_at_cost(
            fixture, calibration_values, 10.0
        ),
        high_cost_environment=_environment_at_cost(
            fixture, calibration_values, 40.0
        ),
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
    assert rebound.outer_inference_plan_receipt_sha256 == _digest(
        "reopened-outer-plan"
    )
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
    assert registry.control_ids == tuple(f"FC{index:02d}" for index in range(8))
    incomplete = SimpleNamespace(
        validate=lambda: None,
        runtime_selection_replayed=True,
        runtime_candidates=(),
    )
    with pytest.raises(
        MassiveAdaptiveRLFixedControlRegistryV1Error,
        match="incomplete",
    ):
        validate_massive_adaptive_rl_fixed_control_registry_coverage_v1(
            registry=registry,
            selection_authority=incomplete,  # type: ignore[arg-type]
        )


def test_outer_evidence_v2_requires_frozen_policy_rollout_authority() -> None:
    authenticated = []
    for index in range(4):
        cost_fold = _fold(index)
        rollout = SimpleNamespace(
            validate=lambda: None,
            fold_index=index,
            frozen_policy_receipt_sha256=(
                cost_fold.frozen_rl_policy_receipt_sha256
            ),
            policy_trace=SimpleNamespace(
                semantic_receipt_sha256=cost_fold.primary_trace_receipt_sha256
            ),
            decision_target_inventory_sha256=(
                cost_fold.decision_target_inventory_sha256
            ),
            semantic_receipt_sha256=_digest(("rollout", index)),
        )
        authority = SimpleNamespace(
            validate=lambda: None,
            runtime_rollout=rollout,
            runtime_rollout_replayed=True,
            outer_evaluation_authorized=False,
            semantic_receipt_sha256=_digest(("rollout-authority", index)),
        )
        cost_ladder = SimpleNamespace(
            outer_rollout_authority_receipt_sha256=(
                authority.semantic_receipt_sha256
            ),
            outer_rollout_receipt_sha256=rollout.semantic_receipt_sha256,
            frozen_policy_receipt_sha256=(
                cost_fold.frozen_rl_policy_receipt_sha256
            ),
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
        authenticated.append(
            build_massive_adaptive_authenticated_rl_outer_fold_v2(
                cost_fold=cost_fold,
                rollout_authority=authority,  # type: ignore[arg-type]
                cost_ladder_authority=cost_authority,  # type: ignore[arg-type]
            )
        )
    evidence = build_massive_adaptive_rl_outer_evidence_v2(authenticated)
    assert evidence.evidence_v1.incremental_rl_log_return_lcb95 > 0.0
    assert not evidence.source_data_qualified

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
