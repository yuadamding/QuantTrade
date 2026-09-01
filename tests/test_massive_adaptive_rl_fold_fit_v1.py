from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rl_quant.evaluation.massive_adaptive_rl_fit_forecast_archive_v1 import (
    materialize_massive_adaptive_rl_fit_forecast_archive_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_fit_inference_plan_v1 import (
    build_massive_adaptive_rl_fit_inference_plan_v1,
)
from rl_quant.features.massive_adaptive_decision_root_v1 import (
    build_massive_adaptive_decision_root_v1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.rl.massive_adaptive_ppo_policy_v1 import (
    MassiveAdaptivePPOActorCriticV1,
)
from rl_quant.training.massive_adaptive_prequential_ppo_runner_v1 import (
    MassiveAdaptivePrequentialPPORunnerV1,
)
from rl_quant.training.massive_adaptive_rl_fit_environment_registry_v1 import (
    build_massive_adaptive_rl_fit_environment_registry_v1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_registry_v1 import (
    registered_massive_adaptive_rl_constant_actions_v1,
)
from rl_quant.training.massive_adaptive_rl_fold_fit_chronology_authority_v1 import (
    build_massive_adaptive_rl_fold_fit_chronology_authority_v1,
)
from rl_quant.training.massive_adaptive_rl_chronology_authority_v1 import (
    is_package_owned_massive_adaptive_rl_fit_chronology_authority_v1,
)
from rl_quant.training.massive_adaptive_rl_training_forecast_authority_v2 import (
    build_massive_adaptive_rl_training_forecast_authority_v2,
)
from rl_quant.workflows import (
    massive_adaptive_rl_runtime_source_reconstruction_v1 as reconstruction,
)
from rl_quant.workflows.massive_adaptive_rl_fold_fit_v1 import (
    MassiveAdaptiveRLFoldFitV1Error,
    run_massive_adaptive_rl_fold_fit_v1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    build_massive_adaptive_rl_experiment_manifest_v3,
)
from test_massive_adaptive_decision_tensor_v1 import _origin
from test_massive_adaptive_forecast_archive_v2 import _expand_context_feature
from test_massive_adaptive_profitability_v1_vertical_slice import (
    _empty_event_archive,
)
from test_massive_adaptive_rl_fit_environment_registry_v1 import (
    _ExactRoleRuntimeGraph,
    _daily_input,
    _fill_source,
    _qualified_fit_block,
    _reseal,
)
from test_massive_adaptive_rl_fit_forecast_v1 import _rl_fit_fixture
from test_massive_adaptive_source_authorized_training_v1 import _context
from test_massive_profitability_v6_vertical_slice import _feature_and_target
from test_massive_trade_replay import _conditions


def _second_fit_block(
    *,
    tmp_path: Path,
    lineage,
    identity,
    sessions,
    split_plan,
):
    (
        _unused_checkpoint,
        _unused_window,
        tensor,
        _unused_roots,
        _unused_plan,
        _unused_split,
        _unused_model,
    ) = _rl_fit_fixture(
        tmp_path / "second-source",
        outer_fold_index=0,
        block_index=1,
        block_sessions=63,
    )
    candidate_dates = tuple(row.session_date for row in sessions.sessions)
    features = []
    origins = []
    contexts = []
    for session_date in tensor.decision_session_dates:
        date_index = candidate_dates.index(session_date)
        history = candidate_dates[date_index - 64 : date_index]
        feature, _target = _feature_and_target(
            decision_session_date=session_date,
            source_session_date=history[-1],
            input_session_dates=history,
            date_index=date_index,
        )
        feature = _expand_context_feature(feature)
        origin = _origin(
            feature,
            action_ids=tuple(row.security_id for row in feature.rows),
            session_authority_receipt_sha256=sessions.receipt_sha256,
        )
        context = _reseal(
            _context(feature, origin),
            identity_authority_receipt_sha256=identity.receipt_sha256,
            source_data_qualified=True,
        )
        features.append(feature)
        origins.append(origin)
        contexts.append(context)
    roots = tuple(
        build_massive_adaptive_decision_root_v1(
            context_origin=context,
            action_origin=origin,
            features=feature,
        )
        for context, origin, feature in zip(
            contexts,
            origins,
            features,
            strict=True,
        )
    )
    plan = build_massive_adaptive_rl_fit_inference_plan_v1(
        decision_tensor=tensor,
        decision_roots=roots,
        split_plan=split_plan,
        outer_fold_index=0,
        block_index=1,
        block_sessions=63,
        model_spec=lineage.model_spec,
    )
    archive = materialize_massive_adaptive_rl_fit_forecast_archive_v1(
        root=tmp_path,
        artifact_id="qualified-fold-fit-block-1",
        checkpoint=lineage.selected_checkpoint,
        training_window_plan=lineage.training_window,
        inference_tensor=tensor,
        inference_decision_roots=roots,
        inference_plan=plan,
        split_plan=split_plan,
        model_spec=lineage.model_spec,
        committed_at_ms=90_000,
    )
    primary_dates = set(plan.origin_session_dates)
    primary_roots = tuple(
        row for row in roots if row.decision_session_date in primary_dates
    )
    primary_contexts = tuple(
        row for row in contexts if row.decision_session_date in primary_dates
    )
    block = reconstruction._fit_block_runtime_sources(
        outer_fold_index=0,
        archive=archive,
        inference_plan=plan,
        lineage=lineage,
        decisions_by_date={row.decision_session_date: row for row in primary_roots},
        contexts_by_date={
            row.decision_session_date: row for row in primary_contexts
        },
    )
    return block


def _fold_fit_runtime_sources(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest = build_massive_adaptive_rl_experiment_manifest_v3(
        experiment_id="package-owned-fold-fit",
        prequential_block_sessions=63,
    )
    block0, lineage, identity, sessions, split_plan = _qualified_fit_block(
        tmp_path / "first",
        block_index=0,
        block_sessions=63,
    )
    block1 = _second_fit_block(
        tmp_path=tmp_path,
        lineage=lineage,
        identity=identity,
        sessions=sessions,
        split_plan=split_plan,
    )
    blocks = (block0, block1)
    roots = tuple(row for block in blocks for row in block.decision_roots)
    contexts = tuple(row for block in blocks for row in block.context_origins)
    fold = reconstruction.MassiveAdaptiveRLFoldRuntimeSourcesV1(
        outer_fold_index=0,
        training_windows=(lineage.training_window,),
        checkpoint_choices=(lineage.checkpoint_choice,),
        calibrations=(lineage.calibration,),
        fit_forecast_archives=tuple(block.forecast_archive for block in blocks),
        decision_roots=roots,
        context_origins=contexts,
        supervised_lineages=(lineage,),
        fit_blocks=blocks,
    )
    fold.validate()
    combined = SimpleNamespace(
        forecast_archive=block0.forecast_archive,
        inference_plan=SimpleNamespace(
            origin_session_dates=tuple(
                date
                for block in blocks
                for date in block.inference_plan.origin_session_dates
            ),
            rows=tuple(
                row for block in blocks for row in block.inference_plan.rows
            ),
        ),
    )
    conditions = _conditions()
    daily = _daily_input(
        block=combined,
        identity=identity,
        sessions=sessions,
        conditions=conditions,
    )
    fill = _fill_source(
        block=combined,
        daily=daily,
        sessions=sessions,
        conditions=conditions,
        qualifying_shares=100.0,
    )
    event_root = tmp_path / "events"
    event_root.mkdir()
    events = _empty_event_archive(
        event_root,
        identity=identity,
        observed_at_ms=max(row.regular_close_at_ms for row in daily.sessions),
    )
    partitions = tuple(
        SimpleNamespace(
            source_session_date=row.source_session_date,
            receipt_sha256=row.persisted_partition_manifest_receipt_sha256,
        )
        for row in daily.sessions
    )
    graph = _ExactRoleRuntimeGraph(
        {
            "session-authority": sessions,
            "condition-authority": conditions,
            "identity-authority": identity,
            "economic-event-archive": events,
            "daily-input-authority": daily,
            "fill-source-authority": fill,
            "split-plan": split_plan,
        }
    )
    monkeypatch.setattr(
        reconstruction.MassiveAdaptiveRLRuntimeSourcesV1,
        "validate",
        lambda _self: None,
    )
    runtime_sources = reconstruction.MassiveAdaptiveRLRuntimeSourcesV1(
        experiment_id=manifest.experiment_id,
        manifest_v3_receipt_sha256=manifest.semantic_receipt_sha256,
        source_bundle_receipt_sha256=semantic_sha256("fold-fit-source-bundle"),
        replay_dependency_index_receipt_sha256=semantic_sha256(
            "fold-fit-dependency-index"
        ),
        runtime_source_graph_authority=graph,  # type: ignore[arg-type]
        session_authority=sessions,
        condition_authority=conditions,
        persisted_partition_manifests=partitions,  # type: ignore[arg-type]
        identity_authority=identity,
        economic_event_archive=events,
        daily_input_authority=daily,
        fill_source=fill,
        split_plan=split_plan,
        folds=(fold,),
        replay_dependency_receipts=(semantic_sha256("fold-fit-dependency"),),
        source_data_qualified=True,
        semantic_receipt_sha256=semantic_sha256("fold-fit-runtime-sources"),
    )
    return manifest, runtime_sources


def test_manifest_v3_registry_executes_genuine_prefix_ppo_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, runtime_sources = _fold_fit_runtime_sources(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    fold = runtime_sources.fold(0)
    training = build_massive_adaptive_rl_training_forecast_authority_v2(
        outer_fold_index=0,
        block_sessions=63,
        split_plan=runtime_sources.split_plan,
        forecast_archives=fold.fit_forecast_archives,
        training_window_plans=fold.training_windows,
        checkpoint_choices=fold.checkpoint_choices,
        calibrations=fold.calibrations,
    )
    registry = build_massive_adaptive_rl_fit_environment_registry_v1(
        manifest=manifest,
        runtime_sources=runtime_sources,
        outer_fold_index=0,
    )
    chronology = build_massive_adaptive_rl_fold_fit_chronology_authority_v1(
        manifest=manifest,
        runtime_sources=runtime_sources,
        training_forecast_authority=training,
    )
    assert is_package_owned_massive_adaptive_rl_fit_chronology_authority_v1(
        chronology
    )
    assert not is_package_owned_massive_adaptive_rl_fit_chronology_authority_v1(
        SimpleNamespace(
            validate=lambda: None,
            fold_index=chronology.fold_index,
            training_forecast_authority_receipt_sha256=(
                chronology.training_forecast_authority_receipt_sha256
            ),
            rl_fit_origin_dates=chronology.rl_fit_origin_dates,
            source_data_qualified=True,
            semantic_receipt_sha256=chronology.semantic_receipt_sha256,
            development_rl_training_authorized=True,
        )
    )
    model = MassiveAdaptivePPOActorCriticV1(observation_dim=90)
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    runner = MassiveAdaptivePrequentialPPORunnerV1(
        training_authority=training,
        chronology_authority=chronology,
        environments=registry.build_environments(),
        fit_environment_authorities={
            receipt: registry.authority(receipt)
            for receipt in registry.forecast_archive_receipts
        },
        model=model,
        config=manifest.base_manifest.ppo_config,
        device="cpu",
    )

    runner.run_next_update()
    checkpoint = runner.checkpoint()

    assert checkpoint.ppo_checkpoint.update_index == 1
    assert checkpoint.source_data_qualified
    assert checkpoint.development_rl_training_authorized
    assert checkpoint.fit_environment_authority_receipts == (
        registry.environment_authorities[0].semantic_receipt_sha256,
    )
    assert checkpoint.transition_decision_session_dates == (
        training.origin_session_dates[:63]
    )
    assert checkpoint.ppo_checkpoint.loss_trace
    assert any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )

    resumed = MassiveAdaptivePrequentialPPORunnerV1(
        training_authority=training,
        chronology_authority=chronology,
        environments=registry.build_environments(),
        fit_environment_authorities={
            receipt: registry.authority(receipt)
            for receipt in registry.forecast_archive_receipts
        },
        model=MassiveAdaptivePPOActorCriticV1(observation_dim=90),
        config=manifest.base_manifest.ppo_config,
        device="cpu",
    )
    resumed.restore(checkpoint)
    assert resumed.current_block_index == 1
    assert resumed.trainer.environment.state.chronology_cursor == 0
    assert resumed.trainer.fit_environment_authority_receipts == [
        registry.environment_authorities[0].semantic_receipt_sha256
    ]
    resumed.trainer.collect_rollout(steps=1)
    assert resumed.trainer.fit_environment_authority_receipts == [
        row.semantic_receipt_sha256
        for row in registry.environment_authorities
    ]


def test_package_owned_fold_fit_executes_ppo_and_complete_fixed_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, runtime_sources = _fold_fit_runtime_sources(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    result = run_massive_adaptive_rl_fold_fit_v1(
        manifest=manifest,
        runtime_sources=runtime_sources,
        outer_fold_index=0,
        artifact_root=tmp_path / "fit-artifacts",
        committed_at_ms=100_000,
        device="cpu",
    )

    result.validate()
    assert result.source_data_qualified
    assert result.development_rl_training_authorized
    assert not result.profitability_reporting_authorized
    assert result.transition_decision_session_dates == (
        result.training_forecast_authority.origin_session_dates
    )
    assert result.candidate_traversed_environment_receipts == (
        tuple(
            row.semantic_receipt_sha256
            for row in result.fit_environment_registry.environment_authorities
        ),
    )
    fixed_run = (
        result.training_workflow.runtime_workflow.fixed_control_fit_authority.runtime_fit_run
    )
    assert fixed_run is not None
    registered_ids = tuple(
        control_id
        for control_id, _action in registered_massive_adaptive_rl_constant_actions_v1()
    )
    assert fixed_run.control_ids == registered_ids
    assert len(fixed_run.traces) == len(registered_ids)
    selection = (
        result.training_workflow.runtime_workflow.fixed_control_selection_authority.runtime_selection
    )
    assert selection is not None
    assert selection.selected_control_id in registered_ids

    wrong_manifest = build_massive_adaptive_rl_experiment_manifest_v3(
        experiment_id="another-fold-fit-experiment",
        prequential_block_sessions=63,
    )
    with pytest.raises(MassiveAdaptiveRLFoldFitV1Error, match="manifest"):
        run_massive_adaptive_rl_fold_fit_v1(
            manifest=wrong_manifest,
            runtime_sources=runtime_sources,
            outer_fold_index=0,
            artifact_root=tmp_path / "wrong-fit-artifacts",
            committed_at_ms=200_000,
            device="cpu",
        )

    overstated = replace(
        result,
        candidate_traversed_environment_receipts=(
            tuple(
                row.semantic_receipt_sha256
                for row in result.fit_environment_registry.environment_authorities
            )
            + (semantic_sha256("future-fit-environment"),),
        ),
        semantic_receipt_sha256="0" * 64,
    )
    overstated = replace(
        overstated,
        candidate_traversed_environment_inventory_sha256=semantic_sha256(
            overstated.candidate_traversed_environment_receipts
        ),
        semantic_receipt_sha256=semantic_sha256(overstated.semantic_unsigned()),
    )
    with pytest.raises(MassiveAdaptiveRLFoldFitV1Error, match="differs"):
        overstated.validate()
