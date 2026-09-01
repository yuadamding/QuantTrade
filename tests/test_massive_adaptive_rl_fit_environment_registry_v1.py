from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_quant.data_sources.massive.source_receipts import (
    load_massive_source_bundle,
    publish_massive_source_object,
)
from rl_quant.evaluation.massive_adaptive_forecast_calibration_v2 import (
    MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V2_DATASET,
    MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V2_SOURCE_SCHEMA_SHA256,
    parse_massive_adaptive_forecast_calibration_v2,
)
from rl_quant.evaluation.massive_adaptive_profitability_env_v1 import (
    MassiveAdaptiveProfitabilityEnvV1,
)
from rl_quant.evaluation.massive_adaptive_rl_fit_forecast_archive_v1 import (
    materialize_massive_adaptive_rl_fit_forecast_archive_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_fit_inference_plan_v1 import (
    build_massive_adaptive_rl_fit_inference_plan_v1,
)
from rl_quant.features.massive_adaptive_decision_root_v1 import (
    build_massive_adaptive_decision_root_v1,
)
from rl_quant.features.massive_adaptive_fill_source_v1 import (
    MASSIVE_ADAPTIVE_FILL_SOURCE_V1_SCHEMA,
    MASSIVE_ADAPTIVE_FILL_SOURCE_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_FILL_SOURCE_V1_SPEC_SHA256,
    MassiveAdaptiveFillRowV1,
    MassiveAdaptiveFillSourceV1,
    adaptive_fill_clock_v1,
)
from rl_quant.features.massive_daily_bars_v0 import MASSIVE_DAILY_BARS_V0_FIELDS
from rl_quant.features.massive_daily_tape_v0 import MASSIVE_DAILY_TAPE_V0_FIELDS
from rl_quant.features.massive_profitability_daily_input_authority_v1 import (
    MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SCHEMA,
    MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SOURCE_SHA256,
    MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SPEC_SHA256,
    MassiveProfitabilityDailyInputAuthorityV1,
    MassiveProfitabilityDailyInputSessionV1,
    MassiveProfitabilityDailySecurityInputV1,
)
from rl_quant.features.massive_profitability_experiment_coverage_v2 import (
    massive_profitability_identity_semantic_receipt_v2,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)
from rl_quant.rl.massive_adaptive_ppo_policy_v1 import (
    MassiveAdaptivePPOActorCriticV1,
)
from rl_quant.rl.massive_adaptive_rl_action_v1 import (
    neutral_massive_adaptive_rl_action_v1,
)
from rl_quant.training.massive_adaptive_ppo_v1 import (
    MassiveAdaptivePPOConfigV1,
    MassiveAdaptivePPOTrainerV1,
    MassiveAdaptivePPOV1Error,
)
from rl_quant.training.massive_adaptive_rl_fit_environment_authority_v1 import (
    MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_AUTHORITY_V1_SCHEMA,
    MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_AUTHORITY_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_AUTHORITY_V1_SPEC_SHA256,
    MassiveAdaptiveRLFitEnvironmentAuthorityV1,
    MassiveAdaptiveRLFitEnvironmentAuthorityV1Error,
    _ISSUER,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_fit_runner_v1 import (
    MassiveAdaptiveRLFixedControlFitRunnerV1Error,
    _bind_block_runtimes,
)
from rl_quant.training.massive_adaptive_rl_training_forecast_authority_v1 import (
    build_massive_adaptive_causal_checkpoint_choice_v1,
)
from rl_quant.workflows import (
    massive_adaptive_rl_runtime_source_reconstruction_v1 as reconstruction,
)
from test_massive_adaptive_decision_tensor_v1 import _origin
from test_massive_adaptive_forecast_archive_v2 import _expand_context_feature
from test_massive_adaptive_profitability_v1_vertical_slice import (
    _calibration_v2,
    _empty_event_archive,
    _identity,
)
from test_massive_adaptive_rl_fit_forecast_v1 import _rl_fit_fixture
from test_massive_adaptive_source_authorized_training_v1 import _context, _sessions
from test_massive_profitability_v6_vertical_slice import _feature_and_target
from test_massive_trade_replay import _conditions


def _reseal(value, **changes):
    provisional = replace(
        value,
        **changes,
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _qualified_fit_block(tmp_path: Path):
    checkpoint, window, tensor, _roots, _plan, split_plan, model_spec = (
        _rl_fit_fixture(
            tmp_path / "fit",
            outer_fold_index=0,
            block_index=0,
            block_sessions=21,
        )
    )
    split_plan = _reseal(split_plan, candidate_source_data_qualified=True)
    window = _reseal(
        window,
        split_plan_receipt_sha256=split_plan.semantic_receipt_sha256,
    )
    checkpoint = _reseal(
        checkpoint,
        split_plan_receipt_sha256=split_plan.semantic_receipt_sha256,
        window_plan_receipt_sha256=window.semantic_receipt_sha256,
        committed_development_training_authorized=True,
        development_training_authorized=True,
    )

    sessions = _sessions()
    candidate_dates = tuple(row.session_date for row in sessions.sessions)
    identity = _identity(tuple(tensor.security_ids))
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
        block_index=0,
        block_sessions=21,
        model_spec=model_spec,
    )
    archive = materialize_massive_adaptive_rl_fit_forecast_archive_v1(
        root=tmp_path,
        artifact_id="qualified-fit-environment",
        checkpoint=checkpoint,
        training_window_plan=window,
        inference_tensor=tensor,
        inference_decision_roots=roots,
        inference_plan=plan,
        split_plan=split_plan,
        model_spec=model_spec,
        committed_at_ms=80_000,
    )
    choice = build_massive_adaptive_causal_checkpoint_choice_v1(
        checkpoints=(checkpoint,),
        training_window_plan=window,
    )

    base = _calibration_v2(tuple(tensor.security_ids)).calibration
    calibration = _reseal(
        base,
        fold_index=0,
        checkpoint_receipt_sha256=checkpoint.semantic_receipt_sha256,
        checkpoint_source_receipt_sha256=checkpoint.loaded_source.receipt_sha256,
        model_state_receipt_sha256=checkpoint.model_state_receipt_sha256,
        training_window_plan_receipt_sha256=window.semantic_receipt_sha256,
        calibration_fit_stop_session_date=choice.selection_cutoff_session_date,
        source_data_qualified=True,
    )
    relative = "massive-adaptive/forecast-calibration-v2/qualified-fit.json"
    publish_massive_source_object(
        stream=BytesIO(
            canonical_json_file_bytes(
                {
                    **calibration.semantic_unsigned(),
                    "semantic_receipt_sha256": (
                        calibration.semantic_receipt_sha256
                    ),
                }
            )
        ),
        root=tmp_path,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V2_DATASET,
        source_object_key=relative,
        requested_at_ms=80_001,
        downloaded_at_ms=80_001,
        schema_sha256=(
            MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V2_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=calibration.fit_population_receipt_sha256,
        committed_at_ms=80_001,
        request_id="QUALIFIED-FIT-CALIBRATION",
    )
    loaded = load_massive_source_bundle(
        root=tmp_path,
        relative_payload_path=relative,
        verified_at_ms=80_001,
    )
    calibration = replace(
        parse_massive_adaptive_forecast_calibration_v2(
            root=tmp_path,
            loaded_source=loaded,
        ),
        runtime_calibration_replayed=True,
        development_calibration_authorized=True,
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
        outer_fold_index=0,
        archive=archive,
        inference_plan=plan,
        lineage=lineage,
        decisions_by_date={row.decision_session_date: row for row in primary_roots},
        contexts_by_date={
            row.decision_session_date: row for row in primary_contexts
        },
    )
    assert block.source_data_qualified
    return block, identity, sessions


def _daily_input(
    *,
    block,
    identity,
    sessions,
    conditions,
) -> MassiveProfitabilityDailyInputAuthorityV1:
    candidate_dates = tuple(row.session_date for row in sessions.sessions)
    first = candidate_dates.index(block.inference_plan.origin_session_dates[0])
    last = candidate_dates.index(block.inference_plan.rows[-1].next_session_date)
    dates = candidate_dates[max(0, first - 63) : last + 1]
    security_ids = tuple(block.forecast_archive.security_ids)
    close_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("close")
    dollar_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("dollar_volume")
    rows = []
    for date_index, session_date in enumerate(dates):
        for security_index, security_id in enumerate(security_ids):
            bars = [1.0] * len(MASSIVE_DAILY_BARS_V0_FIELDS)
            bars[close_index] = 100.0 + security_index + date_index * 0.01
            bars[dollar_index] = 100_000_000.0
            body = {
                "source_session_date": session_date,
                "security_id": security_id,
                "bars_values": tuple(bars),
                "bars_valid": (True,) * len(bars),
                "tape_values": (1.0,) * len(MASSIVE_DAILY_TAPE_V0_FIELDS),
                "tape_valid": (True,) * len(MASSIVE_DAILY_TAPE_V0_FIELDS),
                "signed_dollar_flow": 0.0,
                "same_population_dollar_volume": 1_000_000.0,
                "absolute_signed_flow_imbalance": 0.0,
                "same_population_valid": True,
                "regular_session_event_count": 1,
                "replacement_event_count": 0,
                "cancellation_event_count": 0,
                "late_report_event_count": 0,
                "daily_bar_row_receipt_sha256": semantic_sha256(
                    ("bar", session_date, security_id)
                ),
                "daily_tape_row_receipt_sha256": semantic_sha256(
                    ("tape", session_date, security_id)
                ),
                "tape_population_row_receipt_sha256": semantic_sha256(
                    ("population", session_date, security_id)
                ),
                "persisted_partition_receipt_sha256": semantic_sha256(
                    ("partition", session_date, security_id)
                ),
            }
            row = MassiveProfitabilityDailySecurityInputV1(
                **body,
                receipt_sha256=semantic_sha256(body),
            )
            row.validate()
            rows.append(row)

    session_rows = []
    for session_date in dates:
        fill_start, _fill_end = adaptive_fill_clock_v1(session_date)
        regular_open = fill_start - 5 * 60 * 1_000
        regular_close = fill_start + 6 * 60 * 60 * 1_000
        body = {
            "source_session_date": session_date,
            "regular_open_at_ms": regular_open,
            "regular_close_at_ms": regular_close,
            "vendor_last_modified_at_ms": regular_close,
            "authenticated_get_completed_at_ms": regular_close,
            "authenticated_download_receipt_sha256": semantic_sha256(
                ("download", session_date)
            ),
            "whole_file_scan_receipt_sha256": semantic_sha256(
                ("scan", session_date)
            ),
            "semantic_partition_manifest_receipt_sha256": semantic_sha256(
                ("semantic-partition", session_date)
            ),
            "persisted_partition_manifest_receipt_sha256": semantic_sha256(
                ("persisted-partition", session_date)
            ),
            "daily_bars_artifact_receipt_sha256": semantic_sha256(
                ("bars", session_date)
            ),
            "daily_tape_artifact_receipt_sha256": semantic_sha256(
                ("tape", session_date)
            ),
            "supported_security_row_inventory_sha256": semantic_sha256(
                tuple(
                    row.receipt_sha256
                    for row in rows
                    if row.source_session_date == session_date
                )
            ),
        }
        session_row = MassiveProfitabilityDailyInputSessionV1(
            **body,
            receipt_sha256=semantic_sha256(body),
        )
        session_row.validate()
        session_rows.append(session_row)

    acquisition_receipt = semantic_sha256("qualified-fit-daily-acquisition")
    body = {
        "schema": MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SCHEMA,
        "coverage_start_session_date": dates[0],
        "coverage_end_session_date": dates[-1],
        "data_freeze_at_ms": session_rows[-1].authenticated_get_completed_at_ms,
        "supported_security_ids": security_ids,
        "sessions": tuple(session_rows),
        "rows": tuple(rows),
        "archive_freeze_semantic_receipt_sha256": semantic_sha256(
            "qualified-fit-archive-freeze"
        ),
        "security_support_semantic_receipt_sha256": semantic_sha256(
            "qualified-fit-security-support"
        ),
        "session_authority_receipt_sha256": sessions.receipt_sha256,
        "normalized_identity_semantic_receipt_sha256": (
            massive_profitability_identity_semantic_receipt_v2(identity)
        ),
        "condition_authority_receipt_sha256": conditions.receipt_sha256,
        "correction_authority_receipt_sha256": semantic_sha256(
            "qualified-fit-corrections"
        ),
        "event_domain_spec_receipt_sha256": semantic_sha256(
            "qualified-fit-event-domain"
        ),
        "session_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in session_rows)
        ),
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        "source_transport_qualified": True,
        "daily_input_data_qualified": True,
        "protocol_receipt_sha256": (
            MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
        ),
        "specification_sha256": (
            MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SOURCE_SHA256
        ),
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
    }
    provisional = MassiveProfitabilityDailyInputAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        acquisition_audit_receipt_sha256=acquisition_receipt,
        audit_receipt_sha256="0" * 64,
    )
    semantic = semantic_sha256(provisional.semantic_unsigned())
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": semantic,
                "acquisition_audit_receipt_sha256": acquisition_receipt,
            }
        ),
    )
    result.validate()
    return result


def _fill_source(
    *,
    block,
    daily,
    sessions,
    conditions,
) -> MassiveAdaptiveFillSourceV1:
    dates = tuple(
        dict.fromkeys(row.next_session_date for row in block.inference_plan.rows)
    )
    security_ids = tuple(block.forecast_archive.security_ids)
    rows = []
    for session_date in dates:
        for security_index, security_id in enumerate(security_ids):
            start, end = adaptive_fill_clock_v1(session_date)
            price = 100.005 + security_index
            shares = 1_000_000.0
            body = {
                "session_date": session_date,
                "security_id": security_id,
                "fill_start_at_ms": start,
                "fill_end_at_ms": end,
                "fill_vwap": price,
                "qualifying_share_volume": shares,
                "qualifying_dollar_volume": price * shares,
                "qualifying_trade_count": 10,
                "valid": True,
                "qualifying_trade_inventory_sha256": semantic_sha256(
                    ("qualified-trades", session_date, security_id)
                ),
                "persisted_partition_receipt_sha256": semantic_sha256(
                    ("partition", session_date, security_id)
                ),
                "daily_input_row_receipt_sha256": daily.row(
                    session_date=session_date,
                    security_id=security_id,
                ).receipt_sha256,
            }
            row = MassiveAdaptiveFillRowV1(
                **body,
                receipt_sha256=semantic_sha256(body),
            )
            row.validate()
            rows.append(row)
    body = {
        "schema": MASSIVE_ADAPTIVE_FILL_SOURCE_V1_SCHEMA,
        "session_dates": dates,
        "supported_security_ids": security_ids,
        "rows": tuple(rows),
        "daily_input_authority_semantic_receipt_sha256": (
            daily.semantic_receipt_sha256
        ),
        "session_authority_receipt_sha256": sessions.receipt_sha256,
        "condition_authority_receipt_sha256": conditions.receipt_sha256,
        "persisted_manifest_inventory_sha256": semantic_sha256(
            tuple(("persisted-partition", date) for date in dates)
        ),
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        "source_data_qualified": True,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_FILL_SOURCE_V1_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_FILL_SOURCE_V1_SOURCE_SHA256
        ),
        "source_paths_replayed": True,
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    provisional = MassiveAdaptiveFillSourceV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        audit_receipt_sha256="0" * 64,
    )
    semantic = semantic_sha256(provisional.semantic_unsigned())
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": semantic,
                "daily_input_audit_receipt_sha256": (
                    daily.semantic_receipt_sha256
                ),
            }
        ),
    )
    result.validate()
    return result


def _qualified_environment(tmp_path: Path):
    block, identity, sessions = _qualified_fit_block(tmp_path)
    conditions = _conditions()
    daily = _daily_input(
        block=block,
        identity=identity,
        sessions=sessions,
        conditions=conditions,
    )
    fill = _fill_source(
        block=block,
        daily=daily,
        sessions=sessions,
        conditions=conditions,
    )
    event_root = tmp_path / "events"
    event_root.mkdir()
    events = _empty_event_archive(
        event_root,
        identity=identity,
        observed_at_ms=max(row.regular_close_at_ms for row in daily.sessions),
    )
    environment = MassiveAdaptiveProfitabilityEnvV1(
        forecast_archive=block.forecast_archive,
        calibration=block.calibration,
        inference_plan=block.inference_plan,
        decision_roots=block.decision_roots,
        context_origins=block.context_origins,
        fill_source=fill,
        daily_input_authority=daily,
        identity_authority=identity,
        economic_event_archive=events,
        initial_capital=10_000_000.0,
    )
    return block, environment


def _environment_authority(block, environment):
    plan_dates = tuple(
        row.decision_session_date for row in environment.inference_plan.rows
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_AUTHORITY_V1_SCHEMA,
        "experiment_id": "qualified-fit-environment-test",
        "manifest_v3_receipt_sha256": semantic_sha256("manifest-v3"),
        "runtime_sources_receipt_sha256": semantic_sha256("runtime-sources"),
        "runtime_graph_witness_receipt_sha256": semantic_sha256(
            "runtime-graph-witness"
        ),
        "outer_fold_index": block.outer_fold_index,
        "source_fold_index": block.source_fold_index,
        "block_index": block.block_index,
        "fit_block_receipt_sha256": block.semantic_receipt_sha256,
        "forecast_archive_receipt_sha256": (
            environment.forecast_archive.semantic_receipt_sha256
        ),
        "inference_plan_receipt_sha256": (
            environment.inference_plan.semantic_receipt_sha256
        ),
        "calibration_receipt_sha256": environment.calibration.semantic_receipt_sha256,
        "decision_root_inventory_sha256": semantic_sha256(
            tuple(
                environment.roots[date].semantic_receipt_sha256
                for date in plan_dates
            )
        ),
        "context_origin_inventory_sha256": semantic_sha256(
            tuple(
                environment.contexts[date].semantic_receipt_sha256
                for date in plan_dates
            )
        ),
        "daily_input_authority_receipt_sha256": (
            environment.daily_input_authority.semantic_receipt_sha256
        ),
        "fill_source_receipt_sha256": environment.fill_source.semantic_receipt_sha256,
        "identity_authority_receipt_sha256": (
            environment.identity_authority.receipt_sha256
        ),
        "economic_event_archive_receipt_sha256": (
            environment.economic_event_archive.receipt_sha256
        ),
        "compiler_config_receipt_sha256": environment.compiler_config.receipt_sha256,
        "initial_capital": environment.initial_capital,
        "transaction_cost_basis_points": (
            environment.transaction_cost_basis_points
        ),
        "maximum_fill_participation": environment.maximum_fill_participation,
        "environment_source_inventory_sha256": (
            environment.source_inventory_sha256
        ),
        "economic_compatibility_receipt_sha256": (
            environment.economic_compatibility_receipt_sha256
        ),
        "source_data_qualified": True,
        "runtime_environment_replayed": True,
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_AUTHORITY_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_AUTHORITY_V1_SOURCE_SHA256
        ),
    }
    provisional = MassiveAdaptiveRLFitEnvironmentAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        _runtime_environment=environment,
        _issuer=_ISSUER,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _training_authority(environment):
    block = SimpleNamespace(
        source_forecast_archive_receipt_sha256=(
            environment.forecast_archive.semantic_receipt_sha256
        )
    )
    return SimpleNamespace(
        validate=lambda: None,
        blocks=(block,),
        source_data_qualified=True,
        reinforcement_learning_authorized=True,
        semantic_receipt_sha256=semantic_sha256("fit-training-authority"),
    )


def test_source_qualified_fit_environment_executes_economic_transition(
    tmp_path: Path,
) -> None:
    _block, environment = _qualified_environment(tmp_path)

    observation, _info = environment.reset()
    next_observation, reward, terminated, truncated, info = environment.step(
        neutral_massive_adaptive_rl_action_v1()
    )
    transition = info["transition"]

    assert observation.values
    assert next_observation is not None
    assert not terminated
    assert not truncated
    assert transition.source_data_qualified
    assert (
        transition.economic_step.strategy_execution.economic_event_transition_qualified
    )
    assert (
        transition.economic_step.neutral_execution.economic_event_transition_qualified
    )
    assert reward == pytest.approx(
        10_000.0 * transition.incremental_rl_log_return
    )
    assert not transition.profitability_reporting_authorized


def test_fit_environment_authority_requires_concrete_runtime_witness(
    tmp_path: Path,
) -> None:
    block, environment = _qualified_environment(tmp_path)
    authority = _environment_authority(block, environment)

    with pytest.raises(
        MassiveAdaptiveRLFitEnvironmentAuthorityV1Error,
        match="authority differs",
    ):
        replace(
            authority,
            _runtime_environment=None,
            _issuer=None,
        ).validate()


def test_fixed_controls_bind_the_same_fit_environment_authority(
    tmp_path: Path,
) -> None:
    block, environment = _qualified_environment(tmp_path)
    authority = _environment_authority(block, environment)
    dates = tuple(
        row.decision_session_date for row in environment.inference_plan.rows
    )
    training_block = SimpleNamespace(
        block_index=0,
        semantic_receipt_sha256=semantic_sha256("fixed-control-fit-block"),
        source_forecast_archive_receipt_sha256=(
            environment.forecast_archive.semantic_receipt_sha256
        ),
        calibration_receipt_sha256=environment.calibration.semantic_receipt_sha256,
        forecast_session_dates=dates,
    )
    training_authority = SimpleNamespace(
        blocks=(training_block,),
        outer_fold_index=0,
    )
    receipt = environment.forecast_archive.semantic_receipt_sha256

    runtimes = _bind_block_runtimes(
        training_authority=training_authority,  # type: ignore[arg-type]
        environments={receipt: environment},
        fit_environment_authorities={receipt: authority},
    )

    assert len(runtimes) == 1
    assert (
        runtimes[0].fit_environment_authority_receipt_sha256
        == authority.semantic_receipt_sha256
    )
    with pytest.raises(
        MassiveAdaptiveRLFixedControlFitRunnerV1Error,
        match="registry coverage differs",
    ):
        _bind_block_runtimes(
            training_authority=training_authority,  # type: ignore[arg-type]
            environments={receipt: environment},
            fit_environment_authorities={semantic_sha256("other"): authority},
        )


def test_ppo_rejects_unqualified_transition_after_environment_mutation(
    tmp_path: Path,
) -> None:
    block, environment = _qualified_environment(tmp_path)
    authority = _environment_authority(block, environment)
    trainer = MassiveAdaptivePPOTrainerV1(
        environment=environment,
        model=MassiveAdaptivePPOActorCriticV1(observation_dim=90),
        config=replace(
            MassiveAdaptivePPOConfigV1(),
            rollout_length=1,
            epochs_per_rollout=1,
            minibatch_size=1,
        ),
        training_forecast_authority=_training_authority(environment),
        fit_environment_authority=authority,
    )
    changed_fill = replace(
        environment.fill_source,
        source_data_qualified=False,
        semantic_receipt_sha256="0" * 64,
        audit_receipt_sha256="0" * 64,
    )
    changed_semantic = semantic_sha256(changed_fill.semantic_unsigned())
    changed_fill = replace(
        changed_fill,
        semantic_receipt_sha256=changed_semantic,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": changed_semantic,
                "daily_input_audit_receipt_sha256": (
                    changed_fill.daily_input_authority_semantic_receipt_sha256
                ),
            }
        ),
    )
    changed_fill.validate()
    environment.fill_source = changed_fill

    with pytest.raises(MassiveAdaptivePPOV1Error, match="unqualified"):
        trainer.collect_rollout(steps=1)
