from __future__ import annotations

import inspect
from dataclasses import replace

import pytest

from rl_quant.evaluation.massive_adaptive_rl_fit_forecast_archive_v1 import (
    MassiveAdaptiveRLFitForecastArchiveV1Error,
    authorize_massive_adaptive_rl_fit_forecast_archive_v1,
    materialize_massive_adaptive_rl_fit_forecast_archive_v1,
    parse_massive_adaptive_rl_fit_forecast_archive_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_fit_inference_plan_v1 import (
    MassiveAdaptiveRLFitInferencePlanV1Error,
    build_massive_adaptive_rl_fit_inference_plan_v1,
)
from rl_quant.features.massive_adaptive_decision_root_v1 import (
    build_massive_adaptive_decision_root_v1,
)
from rl_quant.features.massive_adaptive_decision_tensor_v1 import (
    authorize_massive_adaptive_decision_tensor_v1,
    materialize_massive_adaptive_decision_tensor_v1,
    parse_massive_adaptive_decision_tensor_v1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.training.massive_adaptive_split_plan_v1 import (
    build_massive_adaptive_split_plan_v1,
)
from rl_quant.training.massive_adaptive_rl_training_forecast_authority_v1 import (
    build_massive_adaptive_causal_checkpoint_choice_v1,
)
from rl_quant.training.massive_adaptive_rl_training_forecast_authority_v2 import (
    MassiveAdaptiveRLTrainingForecastAuthorityV2Error,
    build_massive_adaptive_rl_training_forecast_authority_v2,
)
from test_massive_adaptive_decision_tensor_v1 import _origin
from test_massive_adaptive_forecast_archive_v1 import _qualified_canary
from test_massive_adaptive_forecast_archive_v2 import _expand_context_feature
from test_massive_adaptive_source_authorized_training_v1 import _context, _sessions
from test_massive_adaptive_profitability_v1_vertical_slice import _calibration_v2
from test_massive_profitability_v6_vertical_slice import _feature_and_target


def _rl_fit_fixture(
    tmp_path,
    *,
    outer_fold_index: int = 0,
    block_index: int = 0,
    block_sessions: int = 21,
):
    (tmp_path / "training").mkdir(parents=True)
    checkpoint, _training_tensor, _training_roots, training_plan, model_spec = (
        _qualified_canary(tmp_path / "training")
    )
    sessions = _sessions()
    candidates = tuple(row.session_date for row in sessions.sessions)
    split_plan = build_massive_adaptive_split_plan_v1(
        candidate_session_dates=candidates,
        session_authority=sessions,
    )
    prefix_count = 126 * (outer_fold_index + 1)
    prefix = split_plan.outer_folds[outer_fold_index].fit_session_dates[-prefix_count:]
    start = block_index * block_sessions
    role_dates = prefix[start : start + block_sessions]
    context_count = model_spec.maximum_context_sessions - 1
    first_index = candidates.index(role_dates[0])
    tensor_dates = candidates[first_index - context_count : first_index] + role_dates
    features = []
    origins = []
    contexts = []
    for session_date in tensor_dates:
        candidate_index = candidates.index(session_date)
        history = candidates[candidate_index - 64 : candidate_index]
        feature, _unused_target = _feature_and_target(
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
        features.append(feature)
        origins.append(origin)
        contexts.append(context)
    artifact_id = f"rl-fit-{outer_fold_index}-{block_index}-{block_sessions}"
    committed = materialize_massive_adaptive_decision_tensor_v1(
        root=tmp_path,
        artifact_id=artifact_id,
        features=tuple(features),
        action_origins=tuple(origins),
        committed_at_ms=60_000,
    )
    generic = parse_massive_adaptive_decision_tensor_v1(
        root=tmp_path, loaded_source=committed.loaded_source
    )
    tensor = authorize_massive_adaptive_decision_tensor_v1(
        root=tmp_path,
        tensor=generic,
        features=tuple(features),
        action_origins=tuple(origins),
    )
    roots = tuple(
        build_massive_adaptive_decision_root_v1(
            context_origin=context,
            action_origin=origin,
            features=feature,
        )
        for context, origin, feature in zip(contexts, origins, features, strict=True)
    )
    plan = build_massive_adaptive_rl_fit_inference_plan_v1(
        decision_tensor=tensor,
        decision_roots=roots,
        split_plan=split_plan,
        outer_fold_index=outer_fold_index,
        block_index=block_index,
        block_sessions=block_sessions,
        model_spec=model_spec,
    )
    return checkpoint, training_plan, tensor, roots, plan, split_plan, model_spec


def test_rl_fit_plan_derives_expanding_prefix_and_exact_block(tmp_path) -> None:
    (
        _checkpoint,
        _training_plan,
        _tensor,
        _roots,
        plan,
        split_plan,
        _model_spec,
    ) = _rl_fit_fixture(
        tmp_path,
        outer_fold_index=3,
        block_index=7,
        block_sessions=63,
    )

    expected_prefix = split_plan.outer_folds[3].fit_session_dates[-504:]
    assert plan.inference_role == "rl_fit"
    assert plan.rl_fit_prefix_session_dates == expected_prefix
    assert plan.origin_session_dates == expected_prefix[441:504]
    assert len(plan.rows) == 63
    assert all(len(row.context_session_dates) == 2 for row in plan.rows)
    assert plan.rl_fit_prefix_inventory_sha256 == semantic_sha256(expected_prefix)
    parameters = inspect.signature(
        build_massive_adaptive_rl_fit_inference_plan_v1
    ).parameters
    assert "target" not in parameters
    assert "inference_role" not in parameters
    assert "origin_session_dates" not in parameters


def test_rl_fit_plan_rejects_nonregistered_block_geometry(tmp_path) -> None:
    (
        _checkpoint,
        _training_plan,
        tensor,
        roots,
        _plan,
        split_plan,
        model_spec,
    ) = _rl_fit_fixture(tmp_path)
    with pytest.raises(MassiveAdaptiveRLFitInferencePlanV1Error, match="21 or 63"):
        build_massive_adaptive_rl_fit_inference_plan_v1(
            decision_tensor=tensor,
            decision_roots=roots,
            split_plan=split_plan,
            outer_fold_index=0,
            block_index=0,
            block_sessions=42,
            model_spec=model_spec,
        )
    with pytest.raises(MassiveAdaptiveRLFitInferencePlanV1Error, match="outside"):
        build_massive_adaptive_rl_fit_inference_plan_v1(
            decision_tensor=tensor,
            decision_roots=roots,
            split_plan=split_plan,
            outer_fold_index=0,
            block_index=6,
            block_sessions=21,
            model_spec=model_spec,
        )


def test_rl_fit_archive_is_target_free_create_only_and_exactly_replayed(
    tmp_path,
) -> None:
    (
        checkpoint,
        training_plan,
        tensor,
        roots,
        plan,
        split_plan,
        model_spec,
    ) = _rl_fit_fixture(tmp_path)
    archive = materialize_massive_adaptive_rl_fit_forecast_archive_v1(
        root=tmp_path,
        artifact_id="rl-fit-forecast-block",
        checkpoint=checkpoint,
        training_window_plan=training_plan,
        inference_tensor=tensor,
        inference_decision_roots=roots,
        inference_plan=plan,
        split_plan=split_plan,
        model_spec=model_spec,
        committed_at_ms=61_000,
    )
    assert archive.runtime_rows is not None
    assert archive.runtime_forecasts_replayed
    assert archive.inference_role == "rl_fit"
    assert archive.origin_session_dates == plan.origin_session_dates
    assert archive.source_fold_index == 0
    assert archive.outer_fold_index == 0
    assert archive.training_tensor_receipt_sha256 != (
        archive.inference_tensor_receipt_sha256
    )
    assert archive.target_maturity_cutoff_session_date < archive.origin_session_dates[0]
    assert not archive.development_forecast_authorized
    assert not archive.reinforcement_learning_authorized

    generic = parse_massive_adaptive_rl_fit_forecast_archive_v1(
        root=tmp_path, loaded_source=archive.loaded_source
    )
    assert generic.runtime_rows is None
    assert not generic.runtime_forecasts_replayed
    replayed = authorize_massive_adaptive_rl_fit_forecast_archive_v1(
        root=tmp_path,
        archive=generic,
        checkpoint=checkpoint,
        training_window_plan=training_plan,
        inference_tensor=tensor,
        inference_decision_roots=roots,
        inference_plan=plan,
        split_plan=split_plan,
        model_spec=model_spec,
    )
    assert replayed.row_receipts == archive.row_receipts

    wrong_window = replace(training_plan, fold_index=1)
    with pytest.raises(
        (MassiveAdaptiveRLFitForecastArchiveV1Error, ValueError),
    ):
        authorize_massive_adaptive_rl_fit_forecast_archive_v1(
            root=tmp_path,
            archive=generic,
            checkpoint=checkpoint,
            training_window_plan=wrong_window,
            inference_tensor=tensor,
            inference_decision_roots=roots,
            inference_plan=plan,
            split_plan=split_plan,
            model_spec=model_spec,
        )


def test_rl_training_authority_v2_consumes_complete_fit_only_prefix(tmp_path) -> None:
    first = _rl_fit_fixture(
        tmp_path / "first",
        outer_fold_index=0,
        block_index=0,
        block_sessions=63,
    )
    second = _rl_fit_fixture(
        tmp_path / "second",
        outer_fold_index=0,
        block_index=1,
        block_sessions=63,
    )
    checkpoint, training_plan, _, _, _, split_plan, model_spec = first
    archives = []
    for index, fixture in enumerate((first, second)):
        _unused_checkpoint, _unused_window, tensor, roots, plan, _, _ = fixture
        archives.append(
            materialize_massive_adaptive_rl_fit_forecast_archive_v1(
                root=tmp_path,
                artifact_id=f"rl-fit-authority-block-{index}",
                checkpoint=checkpoint,
                training_window_plan=training_plan,
                inference_tensor=tensor,
                inference_decision_roots=roots,
                inference_plan=plan,
                split_plan=split_plan,
                model_spec=model_spec,
                committed_at_ms=70_000 + index,
            )
        )
    choice = build_massive_adaptive_causal_checkpoint_choice_v1(
        checkpoints=(checkpoint,),
        training_window_plan=training_plan,
    )
    assert archives[0].runtime_rows is not None
    base_calibration = _calibration_v2(
        archives[0].runtime_rows[0].security_ids
    ).calibration
    calibration = replace(
        base_calibration,
        fold_index=0,
        checkpoint_receipt_sha256=checkpoint.semantic_receipt_sha256,
        checkpoint_source_receipt_sha256=checkpoint.loaded_source.receipt_sha256,
        model_state_receipt_sha256=checkpoint.model_state_receipt_sha256,
        training_window_plan_receipt_sha256=training_plan.semantic_receipt_sha256,
        calibration_fit_stop_session_date=max(
            row.origin_session_date for row in training_plan.rows
        ),
        semantic_receipt_sha256="0" * 64,
    )
    calibration = replace(
        calibration,
        semantic_receipt_sha256=semantic_sha256(calibration.semantic_unsigned()),
    )
    calibration.validate()

    authority = build_massive_adaptive_rl_training_forecast_authority_v2(
        outer_fold_index=0,
        block_sessions=63,
        split_plan=split_plan,
        forecast_archives=tuple(archives),
        training_window_plans=(training_plan,),
        checkpoint_choices=(choice,),
        calibrations=(calibration,),
    )
    expected = split_plan.outer_folds[0].fit_session_dates[-126:]
    assert authority.origin_session_dates == expected
    assert tuple(block.forecast_session_dates for block in authority.blocks) == (
        expected[:63],
        expected[63:],
    )
    assert all(
        max(
            block.supervised_training_cutoff_session_date,
            block.target_maturity_cutoff_session_date,
            block.checkpoint_selection_cutoff_session_date,
            block.calibration_fit_cutoff_session_date,
        )
        < block.forecast_session_dates[0]
        for block in authority.blocks
    )
    assert not authority.source_data_qualified
    assert not authority.reinforcement_learning_authorized

    with pytest.raises(
        MassiveAdaptiveRLTrainingForecastAuthorityV2Error,
        match="incomplete",
    ):
        build_massive_adaptive_rl_training_forecast_authority_v2(
            outer_fold_index=0,
            block_sessions=63,
            split_plan=split_plan,
            forecast_archives=(archives[0],),
            training_window_plans=(training_plan,),
            checkpoint_choices=(choice,),
            calibrations=(calibration,),
        )
