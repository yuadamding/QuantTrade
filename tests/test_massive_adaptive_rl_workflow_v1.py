from __future__ import annotations

import copy
from dataclasses import asdict, replace

import pytest

from rl_quant.evaluation.massive_adaptive_rl_policy_evaluator_v1 import (
    evaluate_massive_adaptive_rl_checkpoint_v1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.training.massive_adaptive_ppo_v1 import MassiveAdaptivePPOConfigV1
from rl_quant.training.massive_adaptive_rl_checkpoint_authority_v1 import (
    materialize_massive_adaptive_rl_checkpoint_authority_v1,
)
from rl_quant.workflows.massive_adaptive_rl_v1 import (
    MassiveAdaptiveRLWorkflowV1Error,
    build_massive_adaptive_rl_experiment_manifest_v1,
    load_massive_adaptive_rl_experiment_manifest_v1,
    main,
    run_massive_adaptive_rl_training_workflow_v1,
    run_massive_adaptive_rl_validation_workflow_v1,
    verify_massive_adaptive_rl_training_workflow_v1,
    write_massive_adaptive_rl_experiment_manifest_v1,
)
from test_massive_adaptive_profitability_v1_vertical_slice import (
    _adaptive_env_fixture,
)
from test_massive_adaptive_rl_authoritative_orchestration_v1 import (
    _chronology,
    _environment_at_cost,
    _training_authority,
)


def _config() -> MassiveAdaptivePPOConfigV1:
    return MassiveAdaptivePPOConfigV1(
        epochs_per_rollout=1,
        rollout_length=1,
        minibatch_size=1,
        seed=17,
    )


def _roots():
    _, _, environment = _adaptive_env_fixture()
    training = _training_authority(environment)
    # The production authority carries this field. The compact two-session
    # fixture predates the package-owned workflow and supplies it explicitly.
    training.block_sessions = 21
    chronology = _chronology(environment, training)
    return environment, training, chronology


def test_adaptive_rl_manifest_is_canonical_create_only_and_duration_free(
    tmp_path,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v1(
        experiment_id="adaptive-rl-canary",
        candidate_update_indices=(1, 2),
        seeds=(17,),
        ppo_config=_config(),
    )
    path = tmp_path / "manifest.json"
    write_massive_adaptive_rl_experiment_manifest_v1(path=path, manifest=manifest)
    reopened = load_massive_adaptive_rl_experiment_manifest_v1(path)

    assert reopened == manifest
    assert not reopened.profitability_reporting_authorized
    assert not reopened.lockbox_access_authorized
    assert reopened.seeds == (17,)
    assert reopened.seed_policy == "canonical-fixed-seed-v1"
    forbidden = ("hold30", "position_age", "duration", "scheduled_exit")
    payload = repr(asdict(reopened)).lower()
    assert not any(value in payload for value in forbidden)
    with pytest.raises(MassiveAdaptiveRLWorkflowV1Error, match="create-only"):
        write_massive_adaptive_rl_experiment_manifest_v1(
            path=path,
            manifest=manifest,
        )

    changed = replace(
        manifest,
        candidate_update_indices=(1, 1),
        semantic_receipt_sha256="0" * 64,
    )
    changed = replace(
        changed,
        semantic_receipt_sha256=semantic_sha256(changed.semantic_unsigned()),
    )
    with pytest.raises(MassiveAdaptiveRLWorkflowV1Error, match="manifest differs"):
        changed.validate()

    with pytest.raises(MassiveAdaptiveRLWorkflowV1Error, match="manifest differs"):
        build_massive_adaptive_rl_experiment_manifest_v1(
            experiment_id="adaptive-rl-multiple-seeds",
            seeds=(17, 23),
            ppo_config=_config(),
        )


def test_package_workflow_publishes_resume_and_policy_checkpoints(tmp_path) -> None:
    environment, training, chronology = _roots()
    manifest = build_massive_adaptive_rl_experiment_manifest_v1(
        experiment_id="dual-checkpoint",
        candidate_update_indices=(1, 2),
        ppo_config=_config(),
    )
    result = run_massive_adaptive_rl_training_workflow_v1(
        manifest=manifest,
        fold_index=0,
        seed=17,
        training_authority=training,
        chronology_authority=chronology,
        environments={
            environment.forecast_archive.semantic_receipt_sha256: environment
        },
        artifact_root=tmp_path,
        committed_at_ms=100,
    )

    assert result.candidate_update_indices == (1, 2)
    assert result.training_run.completed_block_receipts == tuple(
        row.semantic_receipt_sha256 for row in training.blocks
    )
    assert len(result.runner_checkpoint_authorities) == 2
    assert len(result.policy_checkpoint_authorities) == 2
    assert result.fixed_control_fit_authority.runtime_fit_replayed
    assert result.fixed_control_selection_authority.runtime_selection_replayed
    assert tuple(
        row.runtime_checkpoint.ppo_checkpoint.update_index
        for row in result.operational_checkpoint_authorities
        if row.runtime_checkpoint is not None
    ) == (1, 2)
    for runner_authority, policy_authority in zip(
        result.runner_checkpoint_authorities,
        result.policy_checkpoint_authorities,
        strict=True,
    ):
        assert runner_authority.runtime_checkpoint is not None
        assert policy_authority.runtime_checkpoint is not None
        assert (
            runner_authority.runtime_checkpoint.ppo_checkpoint.semantic_receipt_sha256
            == policy_authority.runtime_checkpoint.semantic_receipt_sha256
        )

    first_policy_checkpoint = result.policy_checkpoint_authorities[0].runtime_checkpoint
    assert first_policy_checkpoint is not None
    assert first_policy_checkpoint.loss_trace
    changed_last_loss = list(first_policy_checkpoint.loss_trace[-1])
    changed_last_loss[0] += 1.0
    changed_loss_trace = (
        *first_policy_checkpoint.loss_trace[:-1],
        tuple(changed_last_loss),
    )
    changed_checkpoint = replace(
        first_policy_checkpoint,
        loss_trace=changed_loss_trace,
        loss_trace_receipt_sha256=semantic_sha256(changed_loss_trace),
        semantic_receipt_sha256="0" * 64,
    )
    changed_checkpoint = replace(
        changed_checkpoint,
        semantic_receipt_sha256=semantic_sha256(changed_checkpoint.semantic_unsigned()),
    )
    changed_checkpoint.validate()
    changed_policy = materialize_massive_adaptive_rl_checkpoint_authority_v1(
        root=tmp_path,
        artifact_id="same-update-different-policy",
        checkpoint=changed_checkpoint,
        training_forecast_authority=training,
        committed_at_ms=120,
    )
    changed_policies = (
        changed_policy,
        *result.policy_checkpoint_authorities[1:],
    )
    mismatched_workflow = replace(
        result,
        policy_checkpoint_authorities=changed_policies,
        policy_checkpoint_inventory_sha256=semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in changed_policies)
        ),
        semantic_receipt_sha256="0" * 64,
    )
    mismatched_workflow = replace(
        mismatched_workflow,
        semantic_receipt_sha256=semantic_sha256(
            mismatched_workflow.semantic_unsigned()
        ),
    )
    with pytest.raises(MassiveAdaptiveRLWorkflowV1Error, match="workflow differs"):
        mismatched_workflow.validate()

    missing_policy = (
        tmp_path
        / "massive-adaptive"
        / "rl-checkpoint-v1"
        / "dual-checkpoint-fold0-seed17-update2-policy.pt"
    )
    for path in (
        missing_policy,
        missing_policy.with_name(missing_policy.name + ".receipt.json"),
        missing_policy.with_name(missing_policy.name + ".commit.json"),
    ):
        path.unlink()

    resumed_environment, resumed_training, resumed_chronology = _roots()
    resumed = run_massive_adaptive_rl_training_workflow_v1(
        manifest=manifest,
        fold_index=0,
        seed=17,
        training_authority=resumed_training,
        chronology_authority=resumed_chronology,
        environments={
            resumed_environment.forecast_archive.semantic_receipt_sha256: (
                resumed_environment
            )
        },
        artifact_root=tmp_path,
        committed_at_ms=100,
        resume=True,
    )
    assert resumed.semantic_unsigned() == result.semantic_unsigned()
    assert resumed.semantic_receipt_sha256 == result.semantic_receipt_sha256
    assert resumed.operational_checkpoint_inventory_sha256 == (
        result.operational_checkpoint_inventory_sha256
    )
    assert tuple(
        row.semantic_receipt_sha256
        for row in resumed.operational_checkpoint_authorities
    ) == tuple(
        row.semantic_receipt_sha256
        for row in result.operational_checkpoint_authorities
    )
    assert missing_policy.exists()

    # Validation consumes the workflow-published policy authority, not an
    # independently assembled checkpoint or action sequence.
    _, _, validation_environment = _adaptive_env_fixture()
    evaluated = evaluate_massive_adaptive_rl_checkpoint_v1(
        checkpoint_authority=result.policy_checkpoint_authorities[-1],
        chronology_authority=chronology,  # type: ignore[arg-type]
        environment=validation_environment,
        fold_index=0,
        evaluation_role="inner_validation",
    )
    assert evaluated.checkpoint_receipt_sha256 == (
        result.policy_checkpoint_authorities[-1].checkpoint_receipt_sha256
    )
    assert tuple(
        row.action_receipt_sha256 for row in evaluated.action_evidence
    ) == tuple(row.action_receipt_sha256 for row in evaluated.transitions)

    fixture, calibration, primary = _adaptive_env_fixture()
    cost_environments = (
        _environment_at_cost(fixture, calibration, 10.0),
        primary,
        _environment_at_cost(fixture, calibration, 40.0),
    )
    validation = run_massive_adaptive_rl_validation_workflow_v1(
        manifest=manifest,
        training_workflow=result,
        chronology_authority=chronology,  # type: ignore[arg-type]
        environments={
            authority.semantic_receipt_sha256: cost_environments
            for authority in result.policy_checkpoint_authorities
        },
        fixed_control_environment=_environment_at_cost(fixture, calibration, 20.0),
        artifact_root=tmp_path,
        committed_at_ms=500,
    )
    assert len(validation.policy_trace_authorities) == 2
    assert len(validation.cost_ladder_authorities) == 2
    assert tuple(
        row.policy_trace_receipt_sha256 for row in validation.policy_trace_authorities
    ) == tuple(
        row.primary_trace_receipt_sha256 for row in validation.cost_ladder_authorities
    )
    assert validation.validation_context_receipt_sha256 == (
        primary.validation_context_receipt_sha256
    )
    assert validation.fixed_control_evaluation.policy_trace.evaluation_role == (
        "inner_validation"
    )

    mismatched = copy.copy(primary)
    mismatched.validation_context_receipt_sha256 = semantic_sha256(
        "different-validation-context"
    )
    with pytest.raises(MassiveAdaptiveRLWorkflowV1Error, match="validation context"):
        run_massive_adaptive_rl_validation_workflow_v1(
            manifest=manifest,
            training_workflow=result,
            chronology_authority=chronology,  # type: ignore[arg-type]
            environments={
                authority.semantic_receipt_sha256: (
                    cost_environments
                    if index
                    else (cost_environments[0], mismatched, cost_environments[2])
                )
                for index, authority in enumerate(result.policy_checkpoint_authorities)
            },
            fixed_control_environment=_environment_at_cost(fixture, calibration, 20.0),
            artifact_root=tmp_path / "mismatched",
            committed_at_ms=700,
        )

    strictly_missing = missing_policy.with_name(
        "dual-checkpoint-fold0-seed17-update1-policy.pt"
    )
    strictly_missing_paths = (
        strictly_missing,
        strictly_missing.with_name(strictly_missing.name + ".receipt.json"),
        strictly_missing.with_name(strictly_missing.name + ".commit.json"),
    )
    for path in strictly_missing_paths:
        path.unlink()
    verifying_environment, verifying_training, verifying_chronology = _roots()
    with pytest.raises(
        MassiveAdaptiveRLWorkflowV1Error,
        match="missing a candidate policy checkpoint",
    ):
        verify_massive_adaptive_rl_training_workflow_v1(
            manifest=manifest,
            fold_index=0,
            seed=17,
            training_authority=verifying_training,
            chronology_authority=verifying_chronology,
            environments={
                verifying_environment.forecast_archive.semantic_receipt_sha256: (
                    verifying_environment
                )
            },
            artifact_root=tmp_path,
            verified_at_ms=800,
        )
    assert all(not path.exists() for path in strictly_missing_paths)


def test_package_workflow_rejects_unreached_candidate_update(tmp_path) -> None:
    environment, training, chronology = _roots()
    manifest = build_massive_adaptive_rl_experiment_manifest_v1(
        experiment_id="missing-candidate",
        candidate_update_indices=(1, 3),
        ppo_config=_config(),
    )
    with pytest.raises(
        MassiveAdaptiveRLWorkflowV1Error,
        match="candidate update was not reached",
    ):
        run_massive_adaptive_rl_training_workflow_v1(
            manifest=manifest,
            fold_index=0,
            seed=17,
            training_authority=training,
            chronology_authority=chronology,
            environments={
                environment.forecast_archive.semantic_receipt_sha256: environment
            },
            artifact_root=tmp_path,
            committed_at_ms=200,
        )


def test_adaptive_rl_cli_materializes_and_validates_manifest(tmp_path, capsys) -> None:
    path = tmp_path / "cli-manifest.json"
    assert (
        main(
            [
                "manifest",
                "--experiment-id",
                "cli-canary",
                "--output",
                str(path),
                "--candidate-update",
                "1",
                "--candidate-update",
                "2",
            ]
        )
        == 0
    )
    created_receipt = capsys.readouterr().out.strip()
    assert main(["validate", "--manifest", str(path)]) == 0
    assert capsys.readouterr().out.strip() == created_receipt
