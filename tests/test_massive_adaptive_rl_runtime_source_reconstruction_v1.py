from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rl_quant.alpha.contracts import CorporateActionKind, TerminalEventKind
from rl_quant.data_sources.massive.source_receipts import canonical_json_file_bytes
from rl_quant.evaluation.massive_adaptive_rl_fit_forecast_archive_v1 import (
    materialize_massive_adaptive_rl_fit_forecast_archive_v1,
)
from rl_quant.evaluation.massive_adaptive_profitability_env_v1 import (
    MassiveAdaptiveProfitabilityEnvV1,
)
from rl_quant.evaluation.massive_adaptive_rl_fit_inference_plan_v1 import (
    build_massive_adaptive_rl_fit_inference_plan_v1,
)
from rl_quant.features.massive_adaptive_decision_root_v1 import (
    build_massive_adaptive_decision_root_v1,
)
from rl_quant.features.massive_daily_bars_v0 import MASSIVE_DAILY_BARS_V0_FIELDS
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.training.massive_adaptive_rl_training_forecast_authority_v1 import (
    build_massive_adaptive_causal_checkpoint_choice_v1,
)
from rl_quant.workflows import massive_adaptive_rl_experiment_runner_v2 as runner_module
from rl_quant.workflows import (
    massive_adaptive_rl_runtime_source_reconstruction_v1 as reconstruction,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v2 import (
    run_massive_adaptive_rl_experiment_v2,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    build_massive_adaptive_rl_experiment_manifest_v3,
    write_massive_adaptive_rl_experiment_manifest_v3,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_graph_authority_v1 import (
    authorize_massive_adaptive_rl_runtime_source_graph_authority_v1,
    materialize_massive_adaptive_rl_runtime_source_graph_authority_v1,
)
from test_massive_adaptive_decision_tensor_v1 import _origin
from test_massive_adaptive_forecast_archive_v2 import _expand_context_feature
from test_massive_adaptive_profitability_v1_vertical_slice import (
    _calibration_v2,
    _identity,
)
from test_massive_adaptive_rl_fit_forecast_v1 import _rl_fit_fixture
from test_massive_adaptive_rl_runtime_source_graph_authority_v1 import (
    _allow_synthetic_domain_types,
    _authorized_source_bundle,
)
from test_massive_adaptive_source_authorized_training_v1 import _context, _sessions
from test_massive_economic_authority_v6 import _archive, _corporate, _terminal
from test_massive_trade_replay import _conditions
from test_massive_profitability_v6_vertical_slice import _feature_and_target


def _execution_ready_fit_block(
    tmp_path: Path,
    *,
    outer_fold_index: int = 0,
) -> tuple[
    reconstruction.MassiveAdaptiveRLFitBlockRuntimeSourcesV1,
    reconstruction.MassiveAdaptiveRLSupervisedLineageSourcesV1,
    object,
    object,
]:
    checkpoint, window, tensor, _old_roots, _old_plan, split_plan, model_spec = (
        _rl_fit_fixture(
            tmp_path / "base",
            outer_fold_index=outer_fold_index,
            block_index=0,
            block_sessions=21,
        )
    )
    sessions = _sessions()
    candidate_dates = tuple(row.session_date for row in sessions.sessions)
    identity = _identity(tuple(tensor.security_ids))
    features = []
    origins = []
    contexts = []
    for session_date in tensor.decision_session_dates:
        candidate_index = candidate_dates.index(session_date)
        history = candidate_dates[candidate_index - 64 : candidate_index]
        feature, _target = _feature_and_target(
            decision_session_date=session_date,
            source_session_date=history[-1],
            input_session_dates=history,
            date_index=candidate_index,
        )
        feature = _expand_context_feature(feature)
        action_ids = tuple(row.security_id for row in feature.rows)
        origin = _origin(
            feature,
            action_ids=action_ids,
            session_authority_receipt_sha256=sessions.receipt_sha256,
        )
        context = _context(feature, origin)
        context = replace(
            context,
            identity_authority_receipt_sha256=identity.receipt_sha256,
            semantic_receipt_sha256="0" * 64,
        )
        context = replace(
            context,
            semantic_receipt_sha256=semantic_sha256(context.semantic_unsigned()),
        )
        context.validate()
        features.append(feature)
        origins.append(origin)
        contexts.append(context)
    assert tuple(row.semantic_receipt_sha256 for row in features) == (
        tensor.feature_semantic_receipts
    )
    assert tuple(row.semantic_receipt_sha256 for row in origins) == (
        tensor.action_origin_receipts
    )
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
        outer_fold_index=outer_fold_index,
        block_index=0,
        block_sessions=21,
        model_spec=model_spec,
    )
    archive = materialize_massive_adaptive_rl_fit_forecast_archive_v1(
        root=tmp_path,
        artifact_id=f"execution-ready-fit-{outer_fold_index}",
        checkpoint=checkpoint,
        training_window_plan=window,
        inference_tensor=tensor,
        inference_decision_roots=roots,
        inference_plan=plan,
        split_plan=split_plan,
        model_spec=model_spec,
        committed_at_ms=72_000 + outer_fold_index,
    )
    choice = build_massive_adaptive_causal_checkpoint_choice_v1(
        checkpoints=(checkpoint,),
        training_window_plan=window,
    )
    calibration = _calibration_v2(tuple(tensor.security_ids)).calibration
    calibration = replace(
        calibration,
        fold_index=0,
        checkpoint_receipt_sha256=checkpoint.semantic_receipt_sha256,
        checkpoint_source_receipt_sha256=checkpoint.loaded_source.receipt_sha256,
        model_state_receipt_sha256=checkpoint.model_state_receipt_sha256,
        training_window_plan_receipt_sha256=window.semantic_receipt_sha256,
        calibration_fit_stop_session_date=choice.selection_cutoff_session_date,
        semantic_receipt_sha256="0" * 64,
    )
    calibration = replace(
        calibration,
        semantic_receipt_sha256=semantic_sha256(calibration.semantic_unsigned()),
    )
    calibration.validate()
    objects = {
        checkpoint.semantic_receipt_sha256: checkpoint,
        model_spec.receipt_sha256: model_spec,
        plan.semantic_receipt_sha256: plan,
    }
    lineage = reconstruction._supervised_lineage_runtime_sources(
        source_fold_index=0,
        training_window=window,
        checkpoint_choice=choice,
        calibration=calibration,
        objects=objects,
    )
    primary_dates = set(plan.origin_session_dates)
    primary_roots = tuple(
        row for row in roots if row.decision_session_date in primary_dates
    )
    primary_contexts = tuple(
        row for row in contexts if row.decision_session_date in primary_dates
    )
    block = reconstruction._fit_block_runtime_sources(
        outer_fold_index=outer_fold_index,
        archive=archive,
        inference_plan=plan,
        lineage=lineage,
        decisions_by_date={row.decision_session_date: row for row in primary_roots},
        contexts_by_date={row.decision_session_date: row for row in primary_contexts},
    )
    return block, lineage, identity, sessions


def test_package_snapshot_round_trips_production_authority_and_arrays() -> None:
    session_authority = _sessions()
    payload = reconstruction._snapshot_payload(session_authority)
    restored = reconstruction._parse_snapshot(payload)

    assert restored == session_authority
    encoded = reconstruction._encode_value(
        {
            "tensor": torch.tensor([[1.25, -2.5]], dtype=torch.float32),
            "array": np.asarray([[3, 4]], dtype=np.int64),
        }
    )
    decoded = reconstruction._decode_value(encoded)
    assert isinstance(decoded, dict)
    assert torch.equal(decoded["tensor"], torch.tensor([[1.25, -2.5]]))
    assert np.array_equal(decoded["array"], np.asarray([[3, 4]], dtype=np.int64))


def test_package_snapshot_round_trips_economic_event_enums_through_json(
    tmp_path: Path,
) -> None:
    _identity, archive, _loaded = _archive(
        tmp_path,
        corporate=[
            _corporate(
                provider_event_key="DIV-ENUM",
                provider_revision_id="r0",
                security_id="SEC-A",
                effective_at_ms=2_000,
                available_at_ms=1_500,
                cash_per_share=1.0,
            )
        ],
        terminal=[
            _terminal(
                provider_event_key="WORTHLESS-ENUM",
                security_id="SEC-C",
                effective_at_ms=3_000,
                available_at_ms=2_500,
            )
        ],
        suffix="snapshot-enums",
    )
    payload = reconstruction._snapshot_payload(archive)
    persisted = json.loads(canonical_json_file_bytes(payload))
    restored = reconstruction._parse_snapshot(persisted)

    assert restored == archive
    kinds = tuple(row.source_event.event.kind for row in restored.event_observations)
    assert CorporateActionKind.CASH_DIVIDEND in kinds
    assert TerminalEventKind.WORTHLESS in kinds
    assert all(
        isinstance(kind, CorporateActionKind | TerminalEventKind) for kind in kinds
    )


def test_snapshot_mapping_encoding_is_insertion_order_independent() -> None:
    forward = reconstruction._encode_value({"b": 2, "a": 1})
    reverse = reconstruction._encode_value({"a": 1, "b": 2})

    assert forward == reverse


def test_package_snapshot_rejects_changed_implementation_identity() -> None:
    payload = deepcopy(reconstruction._snapshot_payload(_sessions()))
    encoded = payload["encoded_value"]
    assert isinstance(encoded, dict)
    encoded["implementation_source_sha256"] = semantic_sha256("changed-source")
    body = {
        key: value for key, value in payload.items() if key != "snapshot_receipt_sha256"
    }
    payload["snapshot_receipt_sha256"] = semantic_sha256(body)

    with pytest.raises(
        reconstruction.MassiveAdaptiveRLRuntimeSourceReconstructionV1Error,
        match="implementation differs",
    ):
        reconstruction._parse_snapshot(payload)


def test_dependency_closure_rejects_unreferenced_production_authority() -> None:
    sessions = _sessions()
    primary = (
        (
            "session-authority",
            None,
            "root",
            sessions.receipt_sha256,
            sessions,
        ),
    )

    with pytest.raises(
        reconstruction.MassiveAdaptiveRLRuntimeSourceReconstructionV1Error,
        match="incomplete or contains extras",
    ):
        reconstruction._complete_object_graph(
            primary=primary,
            dependencies=(_conditions(),),
        )


def test_dependency_index_is_create_only_and_rehashes_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v3(
        experiment_id="minimal-replay-dependency-index",
    )
    sessions = _sessions()
    session_spec = reconstruction.MASSIVE_ADAPTIVE_RL_SOURCE_ROLE_REGISTRY_V1[
        "session-authority"
    ]
    monkeypatch.setattr(
        reconstruction,
        "MASSIVE_ADAPTIVE_RL_SOURCE_ROLE_REGISTRY_V1",
        {"session-authority": session_spec},
    )
    graph = SimpleNamespace(
        runtime_authority_receipt_sha256=semantic_sha256("runtime-witness"),
        source_data_qualified=True,
        experiment_id=manifest.experiment_id,
        manifest_v3_receipt_sha256=manifest.semantic_receipt_sha256,
        source_bundle_receipt_sha256=semantic_sha256("source-bundle"),
        semantic_receipt_sha256=semantic_sha256("persisted-graph"),
        validate=lambda: None,
        runtime_authority=lambda **_kwargs: sessions,
    )

    committed = (
        reconstruction.materialize_massive_adaptive_rl_replay_dependency_index_v1(
            source_root=tmp_path,
            manifest=manifest,
            runtime_source_graph_authority=graph,
            replay_dependencies=(),
        )
    )
    loaded = reconstruction.load_massive_adaptive_rl_replay_dependency_index_v1(
        source_root=tmp_path,
        manifest=manifest,
    )
    objects = reconstruction._load_snapshot_objects(root=tmp_path, index=loaded)
    reconstruction._verify_reconstructed_dependency_closure(
        index=loaded,
        objects=objects,
    )

    assert loaded == committed
    assert len(loaded.rows) == 1
    assert loaded.rows[0].semantic_receipt_sha256 == sessions.receipt_sha256
    with pytest.raises(
        reconstruction.MassiveAdaptiveRLRuntimeSourceReconstructionV1Error,
        match="create-only",
    ):
        reconstruction.materialize_massive_adaptive_rl_replay_dependency_index_v1(
            source_root=tmp_path,
            manifest=manifest,
            runtime_source_graph_authority=graph,
            replay_dependencies=(),
        )

    changed_edge = replace(
        loaded.rows[0],
        dependency_receipts=(semantic_sha256("unexpected-dependency"),),
    )
    with pytest.raises(
        reconstruction.MassiveAdaptiveRLRuntimeSourceDependencyMismatch,
        match="closure differs",
    ):
        reconstruction._verify_reconstructed_dependency_closure(
            index=replace(loaded, rows=(changed_edge,)),
            objects=objects,
        )

    snapshot = tmp_path / loaded.rows[0].relative_path
    snapshot.write_bytes(canonical_json_file_bytes({"tampered": True}))
    with pytest.raises(
        reconstruction.MassiveAdaptiveRLRuntimeSourceReconstructionV1Error,
        match="hash differs",
    ):
        reconstruction.load_massive_adaptive_rl_replay_dependency_index_v1(
            source_root=tmp_path,
            manifest=manifest,
        )


def test_rl_fit_dependency_closure_includes_full_tensor_root_inventory(
    tmp_path: Path,
) -> None:
    checkpoint, window, tensor, roots, plan, split_plan, model_spec = _rl_fit_fixture(
        tmp_path,
        outer_fold_index=0,
        block_index=0,
        block_sessions=21,
    )
    for authority in (checkpoint, tensor, plan):
        restored = reconstruction._parse_snapshot(
            reconstruction._snapshot_payload(authority)
        )
        assert type(restored) is type(authority)
        assert reconstruction._receipt(restored) == reconstruction._receipt(authority)
    archive = materialize_massive_adaptive_rl_fit_forecast_archive_v1(
        root=tmp_path,
        artifact_id="reconstruction-fit-forecast",
        checkpoint=checkpoint,
        training_window_plan=window,
        inference_tensor=tensor,
        inference_decision_roots=roots,
        inference_plan=plan,
        split_plan=split_plan,
        model_spec=model_spec,
        committed_at_ms=71_000,
    )
    replayed_archive = reconstruction._parse_snapshot(
        reconstruction._snapshot_payload(archive)
    )
    assert type(replayed_archive) is type(archive)
    assert reconstruction._receipt(replayed_archive) == archive.semantic_receipt_sha256
    root_by_date = {row.decision_session_date: row for row in roots}
    dependencies = reconstruction._expected_dependencies(
        value=archive,
        objects_by_receipt={tensor.semantic_receipt_sha256: tensor},
        primary_decisions_by_date=root_by_date,
    )

    expected_root_receipts = {
        root_by_date[date].semantic_receipt_sha256
        for date in tensor.decision_session_dates
    }
    assert expected_root_receipts <= set(dependencies)
    assert len(expected_root_receipts) > len(archive.origin_session_dates)


def test_reconstruction_uses_auxiliary_decision_roots_for_native_replay(
    tmp_path: Path,
) -> None:
    _checkpoint, _window, tensor, roots, plan, _split_plan, _model_spec = (
        _rl_fit_fixture(
            tmp_path,
            outer_fold_index=0,
            block_index=0,
            block_sessions=21,
        )
    )
    primary_roots = tuple(
        row
        for row in roots
        if row.decision_session_date in plan.rl_fit_prefix_session_dates
    )
    primary_rows = tuple(
        SimpleNamespace(
            role="decision-root-inventory",
            fold_index=0,
            semantic_receipt_sha256=row.semantic_receipt_sha256,
        )
        for row in primary_roots
    )
    objects = {row.semantic_receipt_sha256: row for row in roots}

    by_fold, all_by_date = reconstruction._decision_root_views(
        objects=objects,
        primary_rows=primary_rows,
    )

    assert by_fold[0] == tuple(
        sorted(primary_roots, key=lambda row: row.decision_session_date)
    )
    assert set(tensor.decision_session_dates) <= set(all_by_date)
    assert set(tensor.decision_session_dates) - {
        row.decision_session_date for row in by_fold[0]
    }


def test_fit_block_runtime_sources_reset_real_environment_with_source_fold_lineage(
    tmp_path: Path,
) -> None:
    block, lineage, identity, sessions = _execution_ready_fit_block(
        tmp_path,
        outer_fold_index=1,
    )
    assert block.outer_fold_index == 1
    assert block.source_fold_index == 0
    assert lineage.source_fold_index == 0
    assert block.supervised_lineage_receipt_sha256 == (lineage.semantic_receipt_sha256)
    assert block.inference_plan.origin_session_dates == (
        block.forecast_archive.origin_session_dates
    )
    assert tuple(row.decision_session_date for row in block.decision_roots) == (
        block.forecast_archive.origin_session_dates
    )

    first_date = block.inference_plan.rows[0].decision_session_date
    history_dates = tuple(
        row.session_date for row in sessions.sessions if row.session_date <= first_date
    )[-63:]
    close_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("close")
    dollar_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("dollar_volume")
    bars_rows = {}
    for date_index, session_date in enumerate(history_dates):
        for security_index, security_id in enumerate(
            block.forecast_archive.security_ids
        ):
            values = [0.0] * len(MASSIVE_DAILY_BARS_V0_FIELDS)
            values[close_index] = 100.0 + date_index * 0.1 + security_index * 0.01
            values[dollar_index] = 100_000_000.0
            bars_rows[(session_date, security_id)] = SimpleNamespace(
                bars_values=tuple(values),
                bars_valid=(True,) * len(values),
                receipt_sha256=semantic_sha256(
                    ("fit-daily", session_date, security_id)
                ),
            )
    daily_receipt = semantic_sha256("fit-block-daily-input")
    daily = SimpleNamespace(
        validate=lambda: None,
        sessions=tuple(
            SimpleNamespace(source_session_date=session_date)
            for session_date in history_dates
        ),
        row=lambda *, session_date, security_id: bars_rows[(session_date, security_id)],
        semantic_receipt_sha256=daily_receipt,
        daily_input_data_qualified=False,
    )
    fill_source = SimpleNamespace(
        validate=lambda: None,
        semantic_receipt_sha256=semantic_sha256("fit-block-fill-source"),
        daily_input_authority_semantic_receipt_sha256=daily_receipt,
        source_data_qualified=False,
    )
    environment = MassiveAdaptiveProfitabilityEnvV1(
        forecast_archive=block.forecast_archive,
        calibration=block.calibration,
        inference_plan=block.inference_plan,
        decision_roots=block.decision_roots,
        context_origins=block.context_origins,
        fill_source=fill_source,
        daily_input_authority=daily,
        identity_authority=identity,
        economic_event_archive=None,
        initial_capital=1_000_000.0,
    )

    observation, _info = environment.reset()

    assert observation.values
    assert environment.current_observation == observation
    assert environment._prepared is not None
    assert (
        environment._prepared.strategy_compiler_input_authority.fold_index
        == block.source_fold_index
    )


def test_fold_runtime_sources_expose_canonical_execution_views(
    tmp_path: Path,
) -> None:
    block, lineage, _identity_authority, _session_authority = (
        _execution_ready_fit_block(tmp_path, outer_fold_index=0)
    )
    objects = {
        lineage.selected_checkpoint.semantic_receipt_sha256: (
            lineage.selected_checkpoint
        ),
        lineage.model_spec.receipt_sha256: lineage.model_spec,
        block.inference_plan.semantic_receipt_sha256: block.inference_plan,
    }

    lineages, blocks = reconstruction._fold_execution_runtime_views(
        outer_fold_index=0,
        training_windows=(lineage.training_window,),
        checkpoint_choices=(lineage.checkpoint_choice,),
        calibrations=(lineage.calibration,),
        fit_forecast_archives=(block.forecast_archive,),
        decision_roots=block.decision_roots,
        context_origins=block.context_origins,
        objects=objects,
    )
    fold = reconstruction.MassiveAdaptiveRLFoldRuntimeSourcesV1(
        outer_fold_index=0,
        training_windows=(lineage.training_window,),
        checkpoint_choices=(lineage.checkpoint_choice,),
        calibrations=(lineage.calibration,),
        fit_forecast_archives=(block.forecast_archive,),
        decision_roots=block.decision_roots,
        context_origins=block.context_origins,
        supervised_lineages=lineages,
        fit_blocks=blocks,
    )

    fold.validate()
    assert lineages[0].selected_checkpoint is lineage.selected_checkpoint
    assert blocks[0].inference_plan is block.inference_plan
    assert fold.supervised_lineage(0) is lineages[0]
    assert fold.fit_block(0) is blocks[0]
    assert blocks[0].supervised_lineage_receipt_sha256 == (
        lineages[0].semantic_receipt_sha256
    )
    detached = replace(
        blocks[0],
        supervised_lineage_receipt_sha256=semantic_sha256("detached-lineage"),
        semantic_receipt_sha256="0" * 64,
    )
    detached = replace(
        detached,
        semantic_receipt_sha256=semantic_sha256(detached.semantic_unsigned()),
    )
    detached.validate()
    with pytest.raises(
        reconstruction.MassiveAdaptiveRLRuntimeSourceReconstructionV1Error,
        match="inventory differs",
    ):
        replace(fold, fit_blocks=(detached,)).validate()


def test_dependency_snapshot_path_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(canonical_json_file_bytes({"receipt": semantic_sha256("x")}))
    link = tmp_path / "link.json"
    link.symlink_to(target.name)

    with pytest.raises(
        reconstruction.MassiveAdaptiveRLRuntimeSourceReconstructionV1Error,
        match="symlink",
    ):
        reconstruction._resolve_regular_file(
            root=tmp_path.resolve(),
            relative_path=link.name,
        )

    outside = tmp_path / "outside"
    outside.mkdir()
    parent_link = tmp_path / "reconstruction"
    parent_link.symlink_to(outside.name, target_is_directory=True)
    with pytest.raises(
        reconstruction.MassiveAdaptiveRLRuntimeSourceReconstructionV1Error,
        match="symlink",
    ):
        reconstruction._create_only_output_path(
            root=tmp_path.resolve(),
            relative_path="reconstruction/object.json",
        )


def test_missing_reconstruction_file_is_retryable(tmp_path: Path) -> None:
    with pytest.raises(
        reconstruction.MassiveAdaptiveRLRuntimeSourceTemporarilyUnavailable,
        match="temporarily absent",
    ):
        reconstruction._resolve_regular_file(
            root=tmp_path.resolve(),
            relative_path="missing/object.json",
        )


def test_runner_reconstructs_before_stopping_at_execution_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_synthetic_domain_types(monkeypatch)
    manifest, runtimes, source_bundle = _authorized_source_bundle(
        tmp_path,
        "runner-package-reconstruction",
    )
    materialize_massive_adaptive_rl_runtime_source_graph_authority_v1(
        source_root=tmp_path,
        manifest=manifest,
        source_bundle=source_bundle,
        runtime_sources=runtimes,
    )
    generic_graph = (
        runner_module.load_massive_adaptive_rl_runtime_source_graph_authority_v1(
            source_root=tmp_path,
            manifest=manifest,
            source_bundle=runner_module.load_massive_adaptive_rl_source_bundle_v1(
                source_root=tmp_path,
                manifest=manifest.base_manifest,
            ),
        )
    )
    authorized_graph = authorize_massive_adaptive_rl_runtime_source_graph_authority_v1(
        authority=generic_graph,
        source_bundle=runner_module.load_massive_adaptive_rl_source_bundle_v1(
            source_root=tmp_path,
            manifest=manifest.base_manifest,
        ),
        runtime_sources=runtimes,
    )
    manifest_path = tmp_path / "manifest-v3.json"
    write_massive_adaptive_rl_experiment_manifest_v3(
        path=manifest_path,
        manifest=manifest,
    )
    artifact_root = tmp_path / "artifacts"
    initial = run_massive_adaptive_rl_experiment_v2(
        manifest_path=manifest_path,
        source_root=tmp_path,
        artifact_root=artifact_root,
        device="cpu",
        resume=False,
    )
    assert initial.blocker_code == "runtime-source-replay-dependency-index-required"

    dependency_index = reconstruction.replay_dependency_index_path_v1(
        source_root=tmp_path,
        experiment_id=manifest.experiment_id,
    )
    dependency_index.parent.mkdir(parents=True, exist_ok=True)
    dependency_index.write_bytes(canonical_json_file_bytes({"test": True}))
    runtime_sources = SimpleNamespace(
        runtime_source_graph_authority=authorized_graph,
        semantic_receipt_sha256=semantic_sha256("reconstructed-runtime-sources"),
    )
    monkeypatch.setattr(
        runner_module,
        "reconstruct_and_authorize_massive_adaptive_rl_runtime_sources_v1",
        lambda **_kwargs: runtime_sources,
    )

    result = run_massive_adaptive_rl_experiment_v2(
        manifest_path=manifest_path,
        source_root=tmp_path,
        artifact_root=artifact_root,
        device="cpu",
        resume=True,
    )

    assert result.blocker_code == "four-fold-execution-backend-required"
    assert result.runtime_source_graph_replayed
    assert result.source_data_qualified

    def temporarily_unavailable(**_kwargs: object) -> object:
        raise reconstruction.MassiveAdaptiveRLRuntimeSourceTemporarilyUnavailable(
            "test mount unavailable"
        )

    monkeypatch.setattr(
        runner_module,
        "reconstruct_and_authorize_massive_adaptive_rl_runtime_sources_v1",
        temporarily_unavailable,
    )
    temporarily_blocked = run_massive_adaptive_rl_experiment_v2(
        manifest_path=manifest_path,
        source_root=tmp_path,
        artifact_root=artifact_root,
        device="cpu",
        resume=True,
    )
    repeated = run_massive_adaptive_rl_experiment_v2(
        manifest_path=manifest_path,
        source_root=tmp_path,
        artifact_root=artifact_root,
        device="cpu",
        resume=True,
    )

    assert temporarily_blocked.blocker_code == "runtime-source-temporarily-unavailable"
    assert temporarily_blocked.current_stage.value == "blocked"
    assert repeated.semantic_receipt_sha256 == (
        temporarily_blocked.semantic_receipt_sha256
    )
