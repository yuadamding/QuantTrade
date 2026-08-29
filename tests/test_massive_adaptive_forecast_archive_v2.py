from __future__ import annotations

import inspect
from dataclasses import replace

import pytest
import torch

from rl_quant.evaluation.massive_adaptive_forecast_archive_v2 import (
    MassiveAdaptiveForecastArchiveV2Error,
    authorize_massive_adaptive_forecast_archive_v2,
    materialize_massive_adaptive_forecast_archive_v2,
    parse_massive_adaptive_forecast_archive_v2,
)
from rl_quant.evaluation.massive_adaptive_forecast_eligibility_authority_v2 import (
    MassiveAdaptiveForecastEligibilityAuthorityV2Error,
    build_massive_adaptive_forecast_eligibility_authority_v2,
)
from rl_quant.evaluation.massive_adaptive_forecast_replay_authority_v2 import (
    authorize_massive_adaptive_forecast_replay_authority_v2,
    materialize_massive_adaptive_forecast_replay_authority_v2,
    parse_massive_adaptive_forecast_replay_authority_v2,
)
from rl_quant.evaluation.massive_adaptive_inference_plan_v1 import (
    MassiveAdaptiveInferencePlanV1Error,
    build_massive_adaptive_inference_plan_v1,
)
from rl_quant.features.massive_adaptive_decision_root_v1 import (
    build_massive_adaptive_decision_root_v1,
)
from rl_quant.features.massive_adaptive_decision_tensor_v1 import (
    authorize_massive_adaptive_decision_tensor_v1,
    materialize_massive_adaptive_decision_tensor_v1,
    parse_massive_adaptive_decision_tensor_v1,
)
from rl_quant.features.massive_profitability_origin_features_v2 import (
    MassiveProfitabilityOriginFeatureRowV2,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from test_massive_adaptive_decision_tensor_v1 import _origin
from test_massive_adaptive_forecast_archive_v1 import _qualified_canary
from test_massive_adaptive_source_authorized_training_v1 import _context
from test_massive_profitability_v6_vertical_slice import _feature_and_target


def _expand_context_feature(feature):
    rows = list(feature.rows)
    template = feature.rows[0]
    for asset_index in range(len(rows), 8):
        security_id = f"SEC-{asset_index:02d}"
        bars = list(template.bars_values)
        bars[0] += float(asset_index)
        body = template.unsigned() | {
            "security_id": security_id,
            "decision_membership_rank": asset_index + 1,
            "bars_values": tuple(bars),
            "source_panel_row_receipt_sha256": semantic_sha256(
                (feature.decision_session_date, security_id, "panel")
            ),
            "feature_accounting_security_inventory_sha256": semantic_sha256(
                (feature.decision_session_date, security_id, "feature-accounting")
            ),
            "tape_population_row_receipt_sha256": semantic_sha256(
                (feature.decision_session_date, security_id, "tape-population")
            ),
        }
        rows.append(
            MassiveProfitabilityOriginFeatureRowV2(
                **body,  # type: ignore[arg-type]
                receipt_sha256=semantic_sha256(body),
            )
        )
    changed = replace(
        feature,
        rows=tuple(rows),
        row_inventory_sha256=semantic_sha256(tuple(row.receipt_sha256 for row in rows)),
        semantic_receipt_sha256="0" * 64,
        audit_receipt_sha256=semantic_sha256(
            (feature.decision_session_date, "expanded-feature-audit")
        ),
    )
    result = replace(
        changed,
        semantic_receipt_sha256=semantic_sha256(changed.semantic_unsigned()),
    )
    result.validate()
    return result


def _inner_validation_inference_fixture(tmp_path):
    checkpoint, _training_tensor, _training_roots, training_plan, model_spec = (
        _qualified_canary(tmp_path)
    )
    # Recover the same split geometry used by the package-owned training
    # checkpoint helper. The returned plan is source replayed but deliberately
    # nonauthorizing because this fixture is synthetic.
    from test_massive_adaptive_source_authorized_training_v1 import _sessions
    from rl_quant.training.massive_adaptive_split_plan_v1 import (
        build_massive_adaptive_split_plan_v1,
    )

    sessions = _sessions()
    candidates = tuple(row.session_date for row in sessions.sessions)
    split_plan = build_massive_adaptive_split_plan_v1(
        candidate_session_dates=candidates,
        session_authority=sessions,
    )
    fold = split_plan.outer_folds[0]
    validation_dates = fold.inner_validation_session_dates
    validation_start = candidates.index(validation_dates[0])
    tensor_dates = candidates[validation_start - 1 : validation_start] + (
        validation_dates
    )
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

    committed_tensor = materialize_massive_adaptive_decision_tensor_v1(
        root=tmp_path,
        artifact_id="inner-validation-inference",
        features=tuple(features),
        action_origins=tuple(origins),
        committed_at_ms=50_000,
    )
    generic_tensor = parse_massive_adaptive_decision_tensor_v1(
        root=tmp_path,
        loaded_source=committed_tensor.loaded_source,
    )
    inference_tensor = authorize_massive_adaptive_decision_tensor_v1(
        root=tmp_path,
        tensor=generic_tensor,
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
    inference_plan = build_massive_adaptive_inference_plan_v1(
        decision_tensor=inference_tensor,
        decision_roots=roots,
        split_plan=split_plan,
        fold_index=0,
        inference_role="inner_validation",
        model_spec=model_spec,
    )
    return (
        checkpoint,
        training_plan,
        inference_tensor,
        roots,
        inference_plan,
        split_plan,
        model_spec,
    )


def test_target_free_plan_covers_complete_inner_validation_chronology(
    tmp_path,
) -> None:
    (
        checkpoint,
        training_plan,
        inference_tensor,
        roots,
        inference_plan,
        split_plan,
        model_spec,
    ) = _inner_validation_inference_fixture(tmp_path)

    assert len(inference_plan.rows) == 126
    assert len(inference_tensor.decision_session_dates) == 127
    assert all(len(row.context_session_dates) == 2 for row in inference_plan.rows)
    assert tuple(row.decision_session_date for row in inference_plan.rows) == (
        split_plan.outer_folds[0].inner_validation_session_dates
    )
    assert inference_plan.decision_tensor_receipt_sha256 != (
        checkpoint.decision_tensor_receipt_sha256
    )
    assert inference_plan.origin_decision_root_inventory_sha256 != (
        checkpoint.origin_decision_root_inventory_sha256
    )
    assert set(row.origin_session_date for row in training_plan.rows).isdisjoint(
        row.decision_session_date for row in inference_plan.rows
    )
    assert (
        "target"
        not in inspect.signature(build_massive_adaptive_inference_plan_v1).parameters
    )

    eligibility = build_massive_adaptive_forecast_eligibility_authority_v2(
        checkpoint=checkpoint,
        training_window_plan=training_plan,
        inference_tensor=inference_tensor,
        inference_decision_roots=roots,
        inference_plan=inference_plan,
        model_spec=model_spec,
    )
    assert eligibility.runtime_eligibility_replayed
    assert eligibility.inference_role == "inner_validation"
    assert not eligibility.source_data_qualified
    assert not eligibility.development_forecast_authorized
    assert not eligibility.profitability_reporting_authorized
    assert not eligibility.reinforcement_learning_authorized


def test_v2_archive_and_replay_authority_reexecute_disjoint_inference(
    tmp_path,
) -> None:
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
        artifact_id="inner-validation-forecast",
        checkpoint=checkpoint,
        training_window_plan=training_plan,
        inference_tensor=inference_tensor,
        inference_decision_roots=roots,
        inference_plan=inference_plan,
        model_spec=model_spec,
        committed_at_ms=51_000,
    )
    assert archive.runtime_forecasts_replayed
    assert archive.runtime_rows is not None
    assert len(archive.runtime_rows) == 126
    assert archive.origin_session_dates == tuple(
        row.decision_session_date for row in inference_plan.rows
    )
    assert archive.training_tensor_receipt_sha256 != (
        archive.inference_tensor_receipt_sha256
    )
    assert not archive.development_forecast_authorized
    assert not archive.profitability_reporting_authorized
    assert not archive.lockbox_access_authorized
    assert not archive.reinforcement_learning_authorized
    assert all(torch.isfinite(row.residual_mean).all() for row in archive.runtime_rows)

    generic = parse_massive_adaptive_forecast_archive_v2(
        root=tmp_path, loaded_source=archive.loaded_source
    )
    assert generic.runtime_rows is None
    assert not generic.runtime_forecasts_replayed
    assert not generic.development_forecast_authorized
    replayed = authorize_massive_adaptive_forecast_archive_v2(
        root=tmp_path,
        archive=generic,
        checkpoint=checkpoint,
        training_window_plan=training_plan,
        inference_tensor=inference_tensor,
        inference_decision_roots=roots,
        inference_plan=inference_plan,
        model_spec=model_spec,
    )
    assert replayed.row_receipts == archive.row_receipts

    authority = materialize_massive_adaptive_forecast_replay_authority_v2(
        root=tmp_path,
        artifact_id="inner-validation-replay",
        archive=generic,
        checkpoint=checkpoint,
        training_window_plan=training_plan,
        inference_tensor=inference_tensor,
        inference_decision_roots=roots,
        inference_plan=inference_plan,
        model_spec=model_spec,
        committed_at_ms=52_000,
    )
    assert authority.runtime_forecasts_replayed
    assert not authority.development_forecast_authorized
    generic_authority = parse_massive_adaptive_forecast_replay_authority_v2(
        root=tmp_path, loaded_source=authority.loaded_source
    )
    assert not generic_authority.runtime_forecasts_replayed
    promoted = authorize_massive_adaptive_forecast_replay_authority_v2(
        root=tmp_path,
        authority=generic_authority,
        archive=generic,
        checkpoint=checkpoint,
        training_window_plan=training_plan,
        inference_tensor=inference_tensor,
        inference_decision_roots=roots,
        inference_plan=inference_plan,
        model_spec=model_spec,
    )
    assert promoted.semantic_receipt_sha256 == authority.semantic_receipt_sha256


def test_v2_rejects_wrong_role_fold_root_and_checkpoint(tmp_path) -> None:
    (
        checkpoint,
        training_plan,
        inference_tensor,
        roots,
        inference_plan,
        split_plan,
        model_spec,
    ) = _inner_validation_inference_fixture(tmp_path)
    with pytest.raises(MassiveAdaptiveInferencePlanV1Error, match="only target-free"):
        build_massive_adaptive_inference_plan_v1(
            decision_tensor=inference_tensor,
            decision_roots=roots,
            split_plan=split_plan,
            fold_index=0,
            inference_role="outer_test",
            model_spec=model_spec,
        )
    with pytest.raises(MassiveAdaptiveInferencePlanV1Error, match="complete"):
        build_massive_adaptive_inference_plan_v1(
            decision_tensor=inference_tensor,
            decision_roots=roots,
            split_plan=split_plan,
            fold_index=1,
            inference_role="inner_validation",
            model_spec=model_spec,
        )

    changed_root = replace(
        roots[0],
        action_identity_source_data_qualified=False,
        source_data_qualified=False,
        semantic_receipt_sha256="0" * 64,
    )
    changed_root = replace(
        changed_root,
        semantic_receipt_sha256=semantic_sha256(changed_root.semantic_unsigned()),
    )
    changed_root.validate()
    with pytest.raises(
        MassiveAdaptiveForecastEligibilityAuthorityV2Error,
        match="incompatible",
    ):
        build_massive_adaptive_forecast_eligibility_authority_v2(
            checkpoint=checkpoint,
            training_window_plan=training_plan,
            inference_tensor=inference_tensor,
            inference_decision_roots=(changed_root, *roots[1:]),
            inference_plan=inference_plan,
            model_spec=model_spec,
        )

    assert checkpoint.runtime_state is not None
    changed_state = dict(checkpoint.runtime_state.model_state)
    name = next(iter(changed_state))
    changed_state[name] = changed_state[name].clone()
    changed_state[name].view(-1)[0] += 1.0
    corrupted_checkpoint = replace(
        checkpoint,
        runtime_state=replace(checkpoint.runtime_state, model_state=changed_state),
    )
    with pytest.raises(ValueError):
        build_massive_adaptive_forecast_eligibility_authority_v2(
            checkpoint=corrupted_checkpoint,
            training_window_plan=training_plan,
            inference_tensor=inference_tensor,
            inference_decision_roots=roots,
            inference_plan=inference_plan,
            model_spec=model_spec,
        )

    archive = materialize_massive_adaptive_forecast_archive_v2(
        root=tmp_path,
        artifact_id="inner-validation-row-corruption",
        checkpoint=checkpoint,
        training_window_plan=training_plan,
        inference_tensor=inference_tensor,
        inference_decision_roots=roots,
        inference_plan=inference_plan,
        model_spec=model_spec,
        committed_at_ms=53_000,
    )
    assert archive.runtime_rows is not None
    bad_row = replace(
        archive.runtime_rows[0],
        residual_mean=archive.runtime_rows[0].residual_mean + 1.0,
    )
    with pytest.raises(MassiveAdaptiveForecastArchiveV2Error, match="row receipt"):
        replace(
            archive,
            runtime_rows=(bad_row, *archive.runtime_rows[1:]),
        ).validate()
