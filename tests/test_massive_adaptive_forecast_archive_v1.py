from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.evaluation.massive_adaptive_forecast_archive_v1 import (
    MassiveAdaptiveForecastArchiveV1Error,
    authorize_massive_adaptive_forecast_archive_v1,
    materialize_massive_adaptive_forecast_archive_v1,
    parse_massive_adaptive_forecast_archive_v1,
)
from rl_quant.evaluation.massive_adaptive_forecast_replay_authority_v1 import (
    authorize_massive_adaptive_forecast_replay_authority_v1,
    materialize_massive_adaptive_forecast_replay_authority_v1,
    parse_massive_adaptive_forecast_replay_authority_v1,
)
from rl_quant.features.massive_adaptive_decision_root_v1 import (
    build_massive_adaptive_decision_root_v1,
)
from rl_quant.features.massive_adaptive_decision_tensor_v1 import (
    authorize_massive_adaptive_decision_tensor_v1,
)
from rl_quant.models.adaptive_alpha_term_structure_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_EXPERT_IDS_V1,
    MASSIVE_ADAPTIVE_ALPHA_MODEL_SPEC_V1,
)
from rl_quant.training.massive_adaptive_supervised_trainer_v1 import (
    MASSIVE_ADAPTIVE_SUPERVISED_TRAINING_CONFIG_V1,
    train_and_publish_massive_adaptive_alpha_canary_v1,
)
from rl_quant.training.massive_adaptive_window_plan_v1 import (
    build_massive_adaptive_window_plan_v1,
)
from test_massive_adaptive_source_authorized_training_v1 import _fixture


def _qualified_canary(tmp_path):
    (
        session_authority,
        split_plan,
        features,
        contexts,
        origins,
        source_targets,
        generic_tensor,
    ) = _fixture(tmp_path)
    model_spec = replace(
        MASSIVE_ADAPTIVE_ALPHA_MODEL_SPEC_V1,
        token_dimension=16,
        fast_window_sessions=2,
        maximum_context_sessions=2,
        maximum_intraday_intervals=4,
        market_latent_count=4,
        attention_heads=4,
        dropout_probability=0.0,
    )
    config = replace(
        MASSIVE_ADAPTIVE_SUPERVISED_TRAINING_CONFIG_V1,
        seed=5,
        scheduler_total_updates=8,
    )
    checkpoint = train_and_publish_massive_adaptive_alpha_canary_v1(
        root=tmp_path,
        artifact_id="forecast-checkpoint",
        decision_tensor=generic_tensor,
        features=features,
        context_origins=contexts,
        action_origins=origins,
        source_targets=source_targets,
        session_authority=session_authority,
        split_plan=split_plan,
        fold_index=0,
        split_role="training",
        updates=1,
        committed_at_ms=40_000,
        model_spec=model_spec,
        config=config,
    )
    tensor = authorize_massive_adaptive_decision_tensor_v1(
        root=tmp_path,
        tensor=generic_tensor,
        features=features,
        action_origins=origins,
    )
    roots = tuple(
        build_massive_adaptive_decision_root_v1(
            context_origin=context,
            action_origin=origin,
            features=feature,
        )
        for context, origin, feature in zip(
            contexts, origins, features, strict=True
        )
    )
    window_plan = build_massive_adaptive_window_plan_v1(
        decision_tensor=tensor,
        decision_roots=roots,
        split_plan=split_plan,
        fold_index=0,
        split_role="training",
    )
    return checkpoint, tensor, roots, window_plan, model_spec


def test_forecast_archive_replays_exact_checkpoint_inference(tmp_path) -> None:
    checkpoint, tensor, roots, window_plan, model_spec = _qualified_canary(
        tmp_path
    )
    archive = materialize_massive_adaptive_forecast_archive_v1(
        root=tmp_path,
        artifact_id="forecast-archive",
        checkpoint=checkpoint,
        decision_tensor=tensor,
        decision_roots=roots,
        window_plan=window_plan,
        model_spec=model_spec,
        committed_at_ms=41_000,
    )

    assert archive.runtime_forecasts_replayed
    assert archive.runtime_rows is not None
    assert len(archive.runtime_rows) == len(window_plan.rows)
    assert archive.origin_session_dates == tuple(
        row.origin_session_date for row in window_plan.rows
    )
    assert not archive.committed_source_data_qualified
    assert not archive.development_forecast_authorized
    assert not archive.profitability_reporting_authorized
    assert not archive.reinforcement_learning_authorized
    for row, window_row in zip(
        archive.runtime_rows, window_plan.rows, strict=True
    ):
        assert row.decision_session_date == window_row.origin_session_date
        assert row.valid.dtype == torch.bool
        assert row.residual_mean.shape == (len(tensor.security_ids), 7)
        assert row.router_weights.shape[-2:] == (
            7,
            len(MASSIVE_ADAPTIVE_ALPHA_EXPERT_IDS_V1),
        )
        assert torch.isfinite(row.residual_mean).all()

    generic = parse_massive_adaptive_forecast_archive_v1(
        root=tmp_path, loaded_source=archive.loaded_source
    )
    assert generic.runtime_rows is None
    assert not generic.runtime_forecasts_replayed
    assert not generic.development_forecast_authorized

    replayed = authorize_massive_adaptive_forecast_archive_v1(
        root=tmp_path,
        archive=generic,
        checkpoint=checkpoint,
        decision_tensor=tensor,
        decision_roots=roots,
        window_plan=window_plan,
        model_spec=model_spec,
    )
    assert replayed.row_receipts == archive.row_receipts

    authority = materialize_massive_adaptive_forecast_replay_authority_v1(
        root=tmp_path,
        artifact_id="forecast-replay",
        archive=generic,
        checkpoint=checkpoint,
        decision_tensor=tensor,
        decision_roots=roots,
        window_plan=window_plan,
        model_spec=model_spec,
        committed_at_ms=42_000,
    )
    assert authority.runtime_forecasts_replayed
    assert not authority.development_forecast_authorized
    assert not authority.profitability_reporting_authorized
    assert not authority.reinforcement_learning_authorized

    generic_authority = parse_massive_adaptive_forecast_replay_authority_v1(
        root=tmp_path, loaded_source=authority.loaded_source
    )
    assert not generic_authority.runtime_forecasts_replayed
    assert not generic_authority.development_forecast_authorized
    promoted_authority = authorize_massive_adaptive_forecast_replay_authority_v1(
        root=tmp_path,
        authority=generic_authority,
        archive=generic,
        checkpoint=checkpoint,
        decision_tensor=tensor,
        decision_roots=roots,
        window_plan=window_plan,
        model_spec=model_spec,
    )
    assert promoted_authority.semantic_receipt_sha256 == (
        authority.semantic_receipt_sha256
    )


def test_forecast_replay_rejects_forecast_and_checkpoint_corruption(
    tmp_path,
) -> None:
    checkpoint, tensor, roots, window_plan, model_spec = _qualified_canary(
        tmp_path
    )
    archive = materialize_massive_adaptive_forecast_archive_v1(
        root=tmp_path,
        artifact_id="forecast-corruption",
        checkpoint=checkpoint,
        decision_tensor=tensor,
        decision_roots=roots,
        window_plan=window_plan,
        model_spec=model_spec,
        committed_at_ms=43_000,
    )
    assert archive.runtime_rows is not None
    corrupted_row = replace(
        archive.runtime_rows[0],
        residual_mean=archive.runtime_rows[0].residual_mean + 1.0,
    )
    with pytest.raises(
        MassiveAdaptiveForecastArchiveV1Error,
        match="row receipt differs",
    ):
        replace(
            archive,
            runtime_rows=(corrupted_row, *archive.runtime_rows[1:]),
        ).validate()

    assert checkpoint.runtime_state is not None
    model_state = dict(checkpoint.runtime_state.model_state)
    parameter_name = next(iter(model_state))
    model_state[parameter_name] = model_state[parameter_name].clone()
    model_state[parameter_name].view(-1)[0] += 1.0
    corrupted_checkpoint = replace(
        checkpoint,
        runtime_state=replace(
            checkpoint.runtime_state,
            model_state=model_state,
        ),
    )
    generic = parse_massive_adaptive_forecast_archive_v1(
        root=tmp_path, loaded_source=archive.loaded_source
    )
    with pytest.raises(ValueError):
        authorize_massive_adaptive_forecast_archive_v1(
            root=tmp_path,
            archive=generic,
            checkpoint=corrupted_checkpoint,
            decision_tensor=tensor,
            decision_roots=roots,
            window_plan=window_plan,
            model_spec=model_spec,
        )
