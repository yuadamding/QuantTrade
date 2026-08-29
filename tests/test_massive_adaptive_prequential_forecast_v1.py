from __future__ import annotations

import inspect
from dataclasses import replace

import pytest

from rl_quant.evaluation.massive_adaptive_forecast_archive_v2 import (
    materialize_massive_adaptive_forecast_archive_v2,
    parse_massive_adaptive_forecast_archive_v2,
)
from rl_quant.evaluation.massive_adaptive_prequential_forecast_archive_v1 import (
    MassiveAdaptivePrequentialForecastArchiveV1Error,
    authorize_massive_adaptive_prequential_forecast_archive_v1,
    materialize_massive_adaptive_prequential_forecast_archive_v1,
    parse_massive_adaptive_prequential_forecast_archive_v1,
)
from rl_quant.evaluation.massive_adaptive_prequential_forecast_plan_v1 import (
    MassiveAdaptivePrequentialForecastPlanV1Error,
    build_massive_adaptive_prequential_forecast_plan_v1,
)
from test_massive_adaptive_forecast_archive_v2 import (
    _inner_validation_inference_fixture,
)


def _source_block(tmp_path):
    (
        checkpoint,
        training_plan,
        inference_tensor,
        roots,
        inference_plan,
        _split_plan,
        model_spec,
    ) = _inner_validation_inference_fixture(tmp_path)
    archive = materialize_massive_adaptive_forecast_archive_v2(
        root=tmp_path,
        artifact_id="prequential-source-fold-00",
        checkpoint=checkpoint,
        training_window_plan=training_plan,
        inference_tensor=inference_tensor,
        inference_decision_roots=roots,
        inference_plan=inference_plan,
        model_spec=model_spec,
        committed_at_ms=61_000,
    )
    return training_plan, archive


def test_prequential_plan_and_archive_replay_target_free_fold_prefix(
    tmp_path,
) -> None:
    training_plan, source_archive = _source_block(tmp_path)
    plan = build_massive_adaptive_prequential_forecast_plan_v1(
        forecast_archives=(source_archive,),
        training_window_plans=(training_plan,),
    )

    assert plan.fold_indices == (0,)
    assert plan.origin_session_dates == source_archive.origin_session_dates
    assert plan.blocks[0].training_cutoff_session_date < plan.origin_session_dates[0]
    assert not plan.development_prequential_forecast_authorized
    assert not plan.checkpoint_selection_authorized
    assert not plan.profitability_reporting_authorized
    assert not plan.outer_evaluation_authorized
    assert not plan.lockbox_access_authorized
    assert not plan.reinforcement_learning_authorized
    assert "target" not in inspect.signature(
        build_massive_adaptive_prequential_forecast_plan_v1
    ).parameters

    archive = materialize_massive_adaptive_prequential_forecast_archive_v1(
        root=tmp_path,
        artifact_id="prequential-fold-prefix",
        plan=plan,
        forecast_archives=(source_archive,),
        committed_at_ms=62_000,
    )
    assert archive.runtime_prequential_forecasts_replayed
    assert archive.runtime_rows is not None
    assert len(archive.runtime_rows) == len(source_archive.origin_session_dates)
    assert archive.row_receipts == source_archive.row_receipts
    assert not archive.development_prequential_forecast_authorized
    assert not archive.reinforcement_learning_authorized

    generic = parse_massive_adaptive_prequential_forecast_archive_v1(
        root=tmp_path, loaded_source=archive.loaded_source
    )
    assert generic.runtime_rows is None
    assert not generic.runtime_prequential_forecasts_replayed
    assert not generic.development_prequential_forecast_authorized
    replayed = authorize_massive_adaptive_prequential_forecast_archive_v1(
        root=tmp_path,
        archive=generic,
        plan=plan,
        forecast_archives=(source_archive,),
    )
    assert replayed.semantic_receipt_sha256 == archive.semantic_receipt_sha256
    assert replayed.row_receipts == archive.row_receipts


def test_prequential_plan_rejects_unreplayed_duplicate_and_mutated_blocks(
    tmp_path,
) -> None:
    training_plan, source_archive = _source_block(tmp_path)
    generic_source = parse_massive_adaptive_forecast_archive_v2(
        root=tmp_path, loaded_source=source_archive.loaded_source
    )
    with pytest.raises(
        MassiveAdaptivePrequentialForecastPlanV1Error,
        match="replayed fold forecast",
    ):
        build_massive_adaptive_prequential_forecast_plan_v1(
            forecast_archives=(generic_source,),
            training_window_plans=(training_plan,),
        )
    with pytest.raises(
        MassiveAdaptivePrequentialForecastPlanV1Error,
        match="replayed fold forecast",
    ):
        build_massive_adaptive_prequential_forecast_plan_v1(
            forecast_archives=(source_archive, source_archive),
            training_window_plans=(training_plan, training_plan),
        )

    plan = build_massive_adaptive_prequential_forecast_plan_v1(
        forecast_archives=(source_archive,),
        training_window_plans=(training_plan,),
    )
    assert source_archive.runtime_rows is not None
    changed_row = replace(
        source_archive.runtime_rows[0],
        residual_mean=source_archive.runtime_rows[0].residual_mean + 1.0,
    )
    changed_source = replace(
        source_archive,
        runtime_rows=(changed_row, *source_archive.runtime_rows[1:]),
    )
    with pytest.raises((ValueError, MassiveAdaptivePrequentialForecastArchiveV1Error)):
        materialize_massive_adaptive_prequential_forecast_archive_v1(
            root=tmp_path,
            artifact_id="prequential-mutated-row",
            plan=plan,
            forecast_archives=(changed_source,),
            committed_at_ms=63_000,
        )
