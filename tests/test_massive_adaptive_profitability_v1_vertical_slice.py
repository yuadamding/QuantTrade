from __future__ import annotations

from dataclasses import fields, replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import rl_quant.evaluation.massive_adaptive_profitability_env_v1 as adaptive_env_module

from rl_quant.alpha.pit_universe import (
    ListingEventRecord,
    PITSecurityUniverseAuthority,
    PITUniverseRuleSpec,
    SourcedSecurityMasterRecord,
    SourcedTickerHistoryRecord,
    UniverseRankInputRecord,
)
from rl_quant.data_sources.massive.source_receipts import (
    load_massive_source_bundle,
    publish_massive_source_object,
)
from rl_quant.evaluation.massive_adaptive_forecast_calibration_v1 import (
    build_massive_adaptive_forecast_calibration_v1,
    materialize_massive_adaptive_forecast_calibration_v1,
    parse_massive_adaptive_forecast_calibration_v1,
)
from rl_quant.evaluation.massive_adaptive_profitability_env_v1 import (
    MASSIVE_ADAPTIVE_PROFITABILITY_ENV_V1_SPEC_SHA256,
    MassiveAdaptiveProfitabilityEnvStateV1,
    MassiveAdaptiveProfitabilityEnvV1,
    MassiveAdaptiveRLTransitionV1,
)
from rl_quant.evaluation.massive_adaptive_forecast_calibration_v2 import (
    MassiveAdaptiveForecastCalibrationV2Error,
    build_massive_adaptive_forecast_calibration_v2,
)
from rl_quant.evaluation.massive_adaptive_profit_trace_v1 import (
    build_massive_adaptive_profit_trace_v1,
    materialize_massive_adaptive_profit_trace_v1,
    parse_massive_adaptive_profit_trace_v1,
)
from rl_quant.evaluation.massive_adaptive_profit_trace_v2 import (
    build_massive_adaptive_profit_trace_v2,
    full_portfolio_one_way_turnover_v2,
)
from rl_quant.evaluation.massive_adaptive_profitability_authority_v1 import (
    MassiveAdaptiveProfitabilityAuthorityV1Error,
    build_massive_adaptive_profitability_authority_v1,
)
from rl_quant.evaluation.massive_adaptive_initial_book_authority_v1 import (
    build_massive_adaptive_initial_book_authority_v1,
)
from rl_quant.evaluation.massive_adaptive_economic_step_v1 import (
    prepare_massive_adaptive_economic_step_v1,
    settle_massive_adaptive_economic_step_v1,
)
from rl_quant.execution.massive_adaptive_portfolio_compiler_v1 import (
    MassiveAdaptivePortfolioCompilerConfigV1,
    compile_massive_adaptive_portfolio_v1,
)
from rl_quant.execution.massive_adaptive_rl_compiler_control_v1 import (
    compile_massive_adaptive_rl_control_v1,
)
from rl_quant.evaluation.massive_adaptive_outer_inference_plan_v1 import (
    MassiveAdaptiveOuterInferencePlanV1Error,
    build_massive_adaptive_outer_inference_plan_v1,
)
from rl_quant.evaluation.massive_adaptive_outer_forecast_archive_v1 import (
    MassiveAdaptiveOuterForecastArchiveV1Error,
    materialize_massive_adaptive_outer_forecast_archive_v1,
    validate_massive_adaptive_outer_fold_binding_v1,
)
from rl_quant.features.massive_daily_bars_v0 import MASSIVE_DAILY_BARS_V0_FIELDS
from rl_quant.features.massive_adaptive_fill_source_v1 import adaptive_fill_clock_v1
from rl_quant.features.massive_economic_authority_v6 import (
    MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_DATASETS,
    MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_KINDS,
    MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_OBJECT_PREFIX,
    MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_SCHEMA,
    MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_SOURCE_SCHEMA_SHA256,
    MassiveEconomicAuthorityV6Error,
    build_massive_provider_economic_archive_authority_v6,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)
from rl_quant.rl.massive_adaptive_rl_action_v1 import (
    neutral_massive_adaptive_rl_action_v1,
)
from rl_quant.rl.massive_adaptive_rl_observation_v1 import (
    MassiveAdaptiveRLObservationV1,
    MassiveAdaptiveRLTrailingStateV1,
    build_massive_adaptive_rl_observation_v1,
)
from rl_quant.rl.massive_adaptive_ppo_policy_v1 import (
    MassiveAdaptivePPOActorCriticV1,
)
from rl_quant.training.massive_adaptive_ppo_v1 import (
    MassiveAdaptivePPOConfigV1,
    MassiveAdaptivePPOTrainerV1,
    MassiveAdaptivePPOV1Error,
)
from rl_quant.training.massive_adaptive_rl_checkpoint_authority_v1 import (
    authorize_massive_adaptive_rl_checkpoint_authority_v1,
    materialize_massive_adaptive_rl_checkpoint_authority_v1,
    parse_massive_adaptive_rl_checkpoint_authority_v1,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v1 import (
    build_massive_adaptive_rl_policy_trace_v1,
)
from rl_quant.training.massive_adaptive_profit_checkpoint_selection_v2 import (
    build_massive_adaptive_profit_checkpoint_candidate_v2,
    select_massive_adaptive_profit_checkpoint_v2,
)
from rl_quant.training.massive_adaptive_profit_checkpoint_selection_authority_v2 import (
    MassiveAdaptiveProfitCheckpointSelectionAuthorityV2Error,
    authorize_massive_adaptive_profit_checkpoint_selection_authority_v2,
    materialize_massive_adaptive_profit_checkpoint_selection_authority_v2,
    parse_massive_adaptive_profit_checkpoint_selection_authority_v2,
)


def _digest(value: object) -> str:
    return semantic_sha256(value)


def _calibration(security_ids: tuple[str, ...], *, root=None):
    window_receipt = _digest("training-window-plan")
    valid = torch.ones(len(security_ids), dtype=torch.bool)
    prediction = torch.zeros((len(security_ids), 7), dtype=torch.float32)
    scale = torch.full_like(prediction, 0.01)
    forecast_row = SimpleNamespace(
        decision_session_date="2023-12-29",
        security_ids=security_ids,
        residual_mean=prediction,
        residual_scale=scale,
        valid=valid,
        receipt_sha256=_digest("training-forecast-row"),
    )
    target_rows = tuple(
        SimpleNamespace(
            security_id=security_id,
            residual_bucket_returns=tuple(
                0.001 * (bucket + 1) * (1.0 + index / len(security_ids))
                for bucket in range(7)
            ),
            training_valid_by_bucket=(True,) * 7,
            receipt_sha256=_digest(("training-target", security_id)),
        )
        for index, security_id in enumerate(security_ids)
    )
    source_target = SimpleNamespace(
        decision_session_date="2023-12-29",
        targets=SimpleNamespace(
            security_ids=security_ids,
            rows=target_rows,
        ),
    )
    training_forecasts = SimpleNamespace(
        validate=lambda: None,
        runtime_rows=(forecast_row,),
        runtime_forecasts_replayed=True,
        origin_session_dates=("2023-12-29",),
        semantic_receipt_sha256=_digest("training-forecast-archive"),
        committed_source_data_qualified=False,
        window_plan_receipt_sha256=window_receipt,
    )
    training_targets = SimpleNamespace(
        validate=lambda: None,
        runtime_source_targets=(source_target,),
        runtime_roots_replayed=True,
        decision_session_dates=("2023-12-29",),
        semantic_receipt_sha256=_digest("training-target-archive"),
        committed_source_data_qualified=False,
    )
    training_window_plan = SimpleNamespace(
        validate=lambda: None,
        split_role="training",
        semantic_receipt_sha256=window_receipt,
        rows=(SimpleNamespace(origin_session_date="2023-12-29"),),
    )
    if root is not None:
        return materialize_massive_adaptive_forecast_calibration_v1(
            root=root,
            artifact_id="training-calibration",
            training_forecasts=training_forecasts,
            training_targets=training_targets,
            training_window_plan=training_window_plan,
            committed_at_ms=9_000,
        )
    return build_massive_adaptive_forecast_calibration_v1(
        training_forecasts=training_forecasts,
        training_targets=training_targets,
        training_window_plan=training_window_plan,
    )


def _calibration_v2(security_ids: tuple[str, ...]):
    window_receipt = _digest("training-window-plan-v2")
    checkpoint_receipt = _digest("training-checkpoint-v2")
    checkpoint_source = _digest("training-checkpoint-source-v2")
    model_state = _digest("training-model-state-v2")
    valid = torch.ones(len(security_ids), dtype=torch.bool)
    prediction = torch.zeros((len(security_ids), 7), dtype=torch.float32)
    forecast_row = SimpleNamespace(
        decision_session_date="2023-12-29",
        security_ids=security_ids,
        residual_mean=prediction,
        residual_scale=torch.full_like(prediction, 0.01),
        valid=valid,
        receipt_sha256=_digest("training-forecast-row-v2"),
    )
    target_rows = tuple(
        SimpleNamespace(
            security_id=security_id,
            residual_bucket_returns=tuple(
                0.001 * (bucket + 1) * (1.0 + index / len(security_ids))
                for bucket in range(7)
            ),
            training_valid_by_bucket=(True,) * 7,
            receipt_sha256=_digest(("training-target-v2", security_id)),
        )
        for index, security_id in enumerate(security_ids)
    )
    source_target = SimpleNamespace(
        decision_session_date="2023-12-29",
        targets=SimpleNamespace(security_ids=security_ids, rows=target_rows),
    )
    training_forecasts = SimpleNamespace(
        validate=lambda: None,
        runtime_rows=(forecast_row,),
        runtime_forecasts_replayed=True,
        origin_session_dates=("2023-12-29",),
        semantic_receipt_sha256=_digest("training-forecast-archive-v2"),
        committed_source_data_qualified=False,
        window_plan_receipt_sha256=window_receipt,
        checkpoint_receipt_sha256=checkpoint_receipt,
        model_state_receipt_sha256=model_state,
    )
    training_targets = SimpleNamespace(
        validate=lambda: None,
        runtime_source_targets=(source_target,),
        runtime_roots_replayed=True,
        decision_session_dates=("2023-12-29",),
        semantic_receipt_sha256=_digest("training-target-archive-v2"),
        committed_source_data_qualified=False,
    )
    training_window = SimpleNamespace(
        validate=lambda: None,
        split_role="training",
        fold_index=0,
        semantic_receipt_sha256=window_receipt,
        rows=(SimpleNamespace(origin_session_date="2023-12-29"),),
    )
    checkpoint = SimpleNamespace(
        validate=lambda: None,
        window_plan_receipt_sha256=window_receipt,
        semantic_receipt_sha256=checkpoint_receipt,
        model_state_receipt_sha256=model_state,
        loaded_source=SimpleNamespace(receipt_sha256=checkpoint_source),
        development_training_authorized=False,
    )
    calibration = build_massive_adaptive_forecast_calibration_v2(
        checkpoint=checkpoint,
        training_forecasts=training_forecasts,
        training_targets=training_targets,
        training_window_plan=training_window,
    )
    return SimpleNamespace(
        calibration=calibration,
        checkpoint=checkpoint,
        training_forecasts=training_forecasts,
        training_targets=training_targets,
        training_window=training_window,
    )


def _identity(security_ids: tuple[str, ...]) -> PITSecurityUniverseAuthority:
    rule = PITUniverseRuleSpec.build(
        rule_id="adaptive-profitability-v1-test",
        target_size=len(security_ids),
        ranking_lookback_sessions=3,
        ranking_lag_sessions=1,
        minimum_observed_sessions=2,
        minimum_close_price=1.0,
        minimum_average_dollar_volume=0.0,
        rebalance_frequency="monthly",
    )
    masters = tuple(
        SourcedSecurityMasterRecord(
            security_id=security_id,
            issuer_id=f"ISSUER-{index:03d}",
            primary_exchange="XNYS",
            share_class="COMMON",
            security_type="common-stock",
            listing_at_ms=1,
            delisting_at_ms=None,
            successor_security_id=None,
            corporate_action_chain_id=f"CHAIN-{index:03d}",
            identity_source_receipt_sha256=_digest((security_id, "master")),
        )
        for index, security_id in enumerate(security_ids)
    )
    tickers = tuple(
        SourcedTickerHistoryRecord(
            security_id=security_id,
            ticker=f"T{index:03d}",
            valid_from_ms=1,
            valid_to_ms=None,
            available_at_ms=1,
            primary_exchange="XNYS",
            source_receipt_sha256=_digest((security_id, "ticker")),
        )
        for index, security_id in enumerate(security_ids)
    )
    listings = tuple(
        ListingEventRecord(
            event_id=f"LIST-{security_id}",
            security_id=security_id,
            effective_at_ms=1,
            available_at_ms=1,
            primary_exchange="XNYS",
            ticker=f"T{index:03d}",
            source_receipt_sha256=_digest((security_id, "listing")),
        )
        for index, security_id in enumerate(security_ids)
    )
    ranks = tuple(
        UniverseRankInputRecord(
            security_id=security_id,
            effective_at_ms=100,
            effective_session_index=10,
            available_at_ms=99,
            observation_start_ms=1,
            observation_end_ms=98,
            observation_start_session_index=7,
            observation_end_session_index=9,
            observed_session_count=3,
            average_dollar_volume=1_000_000.0,
            close_price=100.0,
            source_receipt_sha256=_digest((security_id, "rank")),
        )
        for security_id in security_ids
    )
    return PITSecurityUniverseAuthority.build(
        rule=rule,
        security_master=masters,
        ticker_history=tickers,
        listing_events=listings,
        delisting_events=(),
        rank_inputs=ranks,
    )


def _empty_event_archive(tmp_path, *, identity, observed_at_ms: int):
    loaded = []
    for role in MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_KINDS:
        request_id = f"request-{role}"
        payload = {
            "schema": MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_SCHEMA,
            "source_kind": role,
            "provider_id": "massive",
            "provider_dataset": f"provider-{role}",
            "provider_endpoint": "https://api.massive.example/reference",
            "query_start_at_ms": 0,
            "query_end_at_ms": observed_at_ms - 1,
            "provider_observed_at_ms": observed_at_ms,
            "provider_request_ids": [request_id],
            "pagination_complete": True,
            "page_count": 1,
            "records": [],
        }
        relative = (
            f"{MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_OBJECT_PREFIX}"
            f"{role}-adaptive-profitability.json"
        )
        publish_massive_source_object(
            stream=BytesIO(canonical_json_file_bytes(payload)),
            root=tmp_path,
            relative_payload_path=relative,
            dataset_id=MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_DATASETS[role],
            source_object_key=relative,
            requested_at_ms=observed_at_ms - 1,
            downloaded_at_ms=observed_at_ms,
            schema_sha256=(
                MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_SOURCE_SCHEMA_SHA256
            ),
            entitlement_receipt_sha256=_digest("event-entitlement"),
            committed_at_ms=observed_at_ms,
            etag=_digest((role, "empty")),
            request_id=request_id,
        )
        loaded.append(
            load_massive_source_bundle(
                root=tmp_path,
                relative_payload_path=relative,
                verified_at_ms=observed_at_ms,
            )
        )
    return build_massive_provider_economic_archive_authority_v6(
        root=tmp_path,
        loaded_sources=tuple(loaded),
        identity_authority=identity,
    )


def _fixture():
    dates = ("2024-01-02", "2024-01-03", "2024-01-04")
    security_ids = tuple(f"SEC-{index:03d}" for index in range(100))
    identity = _identity(security_ids)
    identity_receipt = identity.receipt_sha256
    bars_rows = {}
    close_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("close")
    dollar_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("dollar_volume")
    for date_index, session_date in enumerate(dates):
        for asset_index, security_id in enumerate(security_ids):
            values = [0.0] * len(MASSIVE_DAILY_BARS_V0_FIELDS)
            price = (
                100.0
                + asset_index * 0.1
                + date_index * (0.2 if asset_index < 50 else -0.1)
            )
            values[close_index] = price
            values[dollar_index] = 100_000_000.0
            bars_rows[(session_date, security_id)] = SimpleNamespace(
                bars_values=tuple(values),
                bars_valid=(True,) * len(values),
                receipt_sha256=_digest(("daily", session_date, security_id)),
            )
    daily_receipt = _digest("daily-input")
    daily = SimpleNamespace(
        validate=lambda: None,
        sessions=tuple(
            SimpleNamespace(
                source_session_date=value,
                regular_close_at_ms=adaptive_fill_clock_v1(value)[1]
                + 6 * 60 * 60 * 1_000,
            )
            for value in dates
        ),
        row=lambda *, session_date, security_id: bars_rows[(session_date, security_id)],
        semantic_receipt_sha256=daily_receipt,
        daily_input_data_qualified=False,
    )
    fill_rows = {}
    for session_date in dates[1:]:
        for asset_index, security_id in enumerate(security_ids):
            price = 100.05 + asset_index * 0.1
            fill_rows[(session_date, security_id)] = SimpleNamespace(
                valid=True,
                fill_vwap=price,
                qualifying_share_volume=1_000_000.0,
                receipt_sha256=_digest(("fill", session_date, security_id)),
            )
    fill_receipt = _digest("fill-source")
    fill_source = SimpleNamespace(
        validate=lambda: None,
        row=lambda *, session_date, security_id: fill_rows[(session_date, security_id)],
        daily_input_authority_semantic_receipt_sha256=daily_receipt,
        semantic_receipt_sha256=fill_receipt,
        source_data_qualified=False,
    )
    forecast_rows = []
    roots = []
    contexts = []
    plan_rows = []
    for date_index, session_date in enumerate(dates[:2]):
        next_session = dates[date_index + 1]
        root_receipt = _digest(("decision-root", session_date))
        context_receipt = _digest(("context", session_date))
        inference_receipt = _digest(("inference-row", session_date))
        means = torch.empty((len(security_ids), 7), dtype=torch.float32)
        for asset_index in range(len(security_ids)):
            means[asset_index] = (0.004 if asset_index < 50 else -0.002) * torch.arange(
                1, 8, dtype=torch.float32
            )
        valid = torch.ones(len(security_ids), dtype=torch.bool)
        forecast_rows.append(
            SimpleNamespace(
                validate=lambda: None,
                decision_session_date=session_date,
                security_ids=security_ids,
                residual_mean=means,
                residual_scale=torch.full_like(means, 0.001),
                valid=valid,
                decision_root_receipt_sha256=root_receipt,
                inference_row_receipt_sha256=inference_receipt,
                array_receipts=(*(_digest((session_date, i)) for i in range(18)),),
                receipt_sha256=_digest(("forecast-row", session_date)),
            )
        )
        roots.append(
            SimpleNamespace(
                validate=lambda: None,
                decision_session_date=session_date,
                context_security_ids=security_ids,
                semantic_receipt_sha256=root_receipt,
                context_origin_receipt_sha256=context_receipt,
                source_data_qualified=False,
            )
        )
        contexts.append(
            SimpleNamespace(
                validate=lambda: None,
                decision_session_date=session_date,
                semantic_receipt_sha256=context_receipt,
                identity_authority_receipt_sha256=identity_receipt,
            )
        )
        plan_rows.append(
            SimpleNamespace(
                validate=lambda **_: None,
                decision_session_date=session_date,
                next_session_date=next_session,
                context_session_dates=(session_date,),
                receipt_sha256=inference_receipt,
            )
        )
    forecast_archive = SimpleNamespace(
        validate=lambda: None,
        runtime_rows=tuple(forecast_rows),
        runtime_forecasts_replayed=True,
        row_receipts=tuple(row.receipt_sha256 for row in forecast_rows),
        semantic_receipt_sha256=_digest("validation-forecast-archive"),
        development_forecast_authorized=False,
    )
    inference_plan = SimpleNamespace(
        validate=lambda: None,
        inference_role="inner_validation",
        rows=tuple(plan_rows),
        semantic_receipt_sha256=_digest("inference-plan"),
    )
    return SimpleNamespace(
        forecast_archive=forecast_archive,
        calibration=_calibration(security_ids),
        inference_plan=inference_plan,
        roots=tuple(roots),
        contexts=tuple(contexts),
        fill_source=fill_source,
        fill_rows=fill_rows,
        daily=daily,
        identity=identity,
    )


def _trace(
    fixture,
    *,
    cost: float = 20.0,
    economic_event_archive=None,
    frozen_decision_trace=None,
):
    return build_massive_adaptive_profit_trace_v1(
        forecast_archive=fixture.forecast_archive,
        calibration=fixture.calibration,
        inference_plan=fixture.inference_plan,
        decision_roots=fixture.roots,
        context_origins=fixture.contexts,
        fill_source=fixture.fill_source,
        daily_input_authority=fixture.daily,
        identity_authority=fixture.identity,
        economic_event_archive=economic_event_archive,
        frozen_decision_trace=frozen_decision_trace,
        initial_capital=10_000_000.0,
        transaction_cost_basis_points=cost,
    )


def test_forecast_to_compiler_to_fill_to_continuous_book_replays(tmp_path) -> None:
    fixture = _fixture()
    trace = _trace(fixture)

    assert len(trace.rows) == 2
    assert trace.rows[1].pretrade_book_receipt_sha256 == (
        trace.rows[0].posttrade_book_receipt_sha256
    )
    assert trace.final_equity > 0.0
    assert trace.final_benchmark_equity > 0.0
    assert trace.deterministic_profitability_replayed
    assert not trace.source_data_qualified
    assert not trace.profitability_reporting_authorized
    assert not trace.reinforcement_learning_authorized

    committed = materialize_massive_adaptive_profit_trace_v1(
        root=tmp_path,
        artifact_id="inner-validation-profit",
        forecast_archive=fixture.forecast_archive,
        calibration=fixture.calibration,
        inference_plan=fixture.inference_plan,
        decision_roots=fixture.roots,
        context_origins=fixture.contexts,
        fill_source=fixture.fill_source,
        daily_input_authority=fixture.daily,
        identity_authority=fixture.identity,
        initial_capital=10_000_000.0,
        committed_at_ms=10_000,
    )
    assert committed.deterministic_profitability_replayed
    assert committed.loaded_source is not None
    generic = parse_massive_adaptive_profit_trace_v1(
        root=tmp_path, loaded_source=committed.loaded_source
    )
    assert not generic.deterministic_profitability_replayed
    assert generic.semantic_receipt_sha256 == committed.semantic_receipt_sha256

    authority = build_massive_adaptive_profitability_authority_v1(
        trace=committed,
        forecast_archive=fixture.forecast_archive,
        calibration=fixture.calibration,
        inference_plan=fixture.inference_plan,
        decision_roots=fixture.roots,
        context_origins=fixture.contexts,
        fill_source=fixture.fill_source,
        daily_input_authority=fixture.daily,
        identity_authority=fixture.identity,
    )
    assert authority.deterministic_trace_replayed
    assert not authority.development_profitability_authorized
    assert not authority.profitability_reporting_authorized
    assert not authority.reinforcement_learning_authorized


def test_cost_ladder_and_fill_mutation_are_economically_visible() -> None:
    fixture = _fixture()
    primary = _trace(fixture, cost=20.0)
    low = _trace(fixture, cost=10.0, frozen_decision_trace=primary)
    high = _trace(fixture, cost=40.0, frozen_decision_trace=primary)
    assert low.final_equity > primary.final_equity > high.final_equity
    assert low.frozen_actions_replayed
    assert high.frozen_actions_replayed
    assert tuple(row.decision_target_receipt_sha256 for row in low.rows) == tuple(
        row.decision_target_receipt_sha256 for row in primary.rows
    )
    assert tuple(row.decision_target_receipt_sha256 for row in high.rows) == tuple(
        row.decision_target_receipt_sha256 for row in primary.rows
    )
    stress_authority = build_massive_adaptive_profitability_authority_v1(
        trace=high,
        forecast_archive=fixture.forecast_archive,
        calibration=fixture.calibration,
        inference_plan=fixture.inference_plan,
        decision_roots=fixture.roots,
        context_origins=fixture.contexts,
        fill_source=fixture.fill_source,
        daily_input_authority=fixture.daily,
        identity_authority=fixture.identity,
        frozen_decision_trace=primary,
    )
    assert stress_authority.deterministic_trace_replayed
    with pytest.raises(MassiveAdaptiveProfitabilityAuthorityV1Error):
        build_massive_adaptive_profitability_authority_v1(
            trace=high,
            forecast_archive=fixture.forecast_archive,
            calibration=fixture.calibration,
            inference_plan=fixture.inference_plan,
            decision_roots=fixture.roots,
            context_origins=fixture.contexts,
            fill_source=fixture.fill_source,
            daily_input_authority=fixture.daily,
            identity_authority=fixture.identity,
        )

    original = primary
    key = next(iter(fixture.fill_rows))
    changed_rows = dict(fixture.fill_rows)
    changed_rows[key] = SimpleNamespace(
        **{
            **changed_rows[key].__dict__,
            "fill_vwap": changed_rows[key].fill_vwap * 1.01,
            "receipt_sha256": _digest(("mutated-fill", key)),
        }
    )
    changed_fill = SimpleNamespace(
        **{
            **fixture.fill_source.__dict__,
            "semantic_receipt_sha256": _digest("mutated-fill-source"),
            "row": lambda *, session_date, security_id: changed_rows[
                (session_date, security_id)
            ],
        }
    )
    with pytest.raises(MassiveAdaptiveProfitabilityAuthorityV1Error):
        build_massive_adaptive_profitability_authority_v1(
            trace=original,
            forecast_archive=fixture.forecast_archive,
            calibration=fixture.calibration,
            inference_plan=fixture.inference_plan,
            decision_roots=fixture.roots,
            context_origins=fixture.contexts,
            fill_source=changed_fill,
            daily_input_authority=fixture.daily,
            identity_authority=fixture.identity,
        )


def test_provider_archive_qualifies_event_complete_book_transition(tmp_path) -> None:
    fixture = _fixture()
    observed_at_ms = max(
        row.regular_close_at_ms for row in fixture.daily.sessions
    )
    archive = _empty_event_archive(
        tmp_path,
        identity=fixture.identity,
        observed_at_ms=observed_at_ms,
    )
    trace = _trace(fixture, economic_event_archive=archive)
    assert trace.economic_event_transition_qualified
    assert trace.economic_event_authority_inventory_sha256 == semantic_sha256(
        (archive.receipt_sha256,)
    )
    assert not trace.source_data_qualified

    authority = build_massive_adaptive_profitability_authority_v1(
        trace=trace,
        forecast_archive=fixture.forecast_archive,
        calibration=fixture.calibration,
        inference_plan=fixture.inference_plan,
        decision_roots=fixture.roots,
        context_origins=fixture.contexts,
        fill_source=fixture.fill_source,
        daily_input_authority=fixture.daily,
        identity_authority=fixture.identity,
        economic_event_archive=archive,
    )
    assert authority.deterministic_trace_replayed

    with pytest.raises(MassiveEconomicAuthorityV6Error, match="archive|receipt"):
        build_massive_adaptive_profitability_authority_v1(
            trace=trace,
            forecast_archive=fixture.forecast_archive,
            calibration=fixture.calibration,
            inference_plan=fixture.inference_plan,
            decision_roots=fixture.roots,
            context_origins=fixture.contexts,
            fill_source=fixture.fill_source,
            daily_input_authority=fixture.daily,
            identity_authority=fixture.identity,
            economic_event_archive=replace(archive, receipt_sha256="f" * 64),
        )


def test_checkpoint_selection_v2_derives_frozen_action_economics(tmp_path) -> None:
    fixture = _fixture()
    primary = _trace(fixture, cost=20.0)
    low = _trace(fixture, cost=10.0, frozen_decision_trace=primary)
    high = _trace(fixture, cost=40.0, frozen_decision_trace=primary)
    primary_authority = build_massive_adaptive_profitability_authority_v1(
        trace=primary,
        forecast_archive=fixture.forecast_archive,
        calibration=fixture.calibration,
        inference_plan=fixture.inference_plan,
        decision_roots=fixture.roots,
        context_origins=fixture.contexts,
        fill_source=fixture.fill_source,
        daily_input_authority=fixture.daily,
        identity_authority=fixture.identity,
    )
    low_authority = build_massive_adaptive_profitability_authority_v1(
        trace=low,
        forecast_archive=fixture.forecast_archive,
        calibration=fixture.calibration,
        inference_plan=fixture.inference_plan,
        decision_roots=fixture.roots,
        context_origins=fixture.contexts,
        fill_source=fixture.fill_source,
        daily_input_authority=fixture.daily,
        identity_authority=fixture.identity,
        frozen_decision_trace=primary,
    )
    high_authority = build_massive_adaptive_profitability_authority_v1(
        trace=high,
        forecast_archive=fixture.forecast_archive,
        calibration=fixture.calibration,
        inference_plan=fixture.inference_plan,
        decision_roots=fixture.roots,
        context_origins=fixture.contexts,
        fill_source=fixture.fill_source,
        daily_input_authority=fixture.daily,
        identity_authority=fixture.identity,
        frozen_decision_trace=primary,
    )
    checkpoint_receipt = _digest("candidate-checkpoint")
    checkpoint_source = _digest("candidate-checkpoint-source")
    model_state = _digest("candidate-model-state")
    checkpoint = SimpleNamespace(
        validate=lambda: None,
        epoch_index=3,
        semantic_receipt_sha256=checkpoint_receipt,
        loaded_source=SimpleNamespace(receipt_sha256=checkpoint_source),
        model_state_receipt_sha256=model_state,
        development_training_authorized=False,
    )
    fixture.forecast_archive.checkpoint_receipt_sha256 = checkpoint_receipt
    fixture.forecast_archive.checkpoint_source_receipt_sha256 = checkpoint_source
    fixture.forecast_archive.model_state_receipt_sha256 = model_state
    candidate = build_massive_adaptive_profit_checkpoint_candidate_v2(
        checkpoint=checkpoint,
        forecast_archive=fixture.forecast_archive,
        primary_trace=primary,
        low_cost_trace=low,
        high_cost_trace=high,
        primary_authority=primary_authority,
        low_cost_authority=low_authority,
        high_cost_authority=high_authority,
    )
    assert candidate.economically_eligible
    assert not candidate.source_data_qualified
    assert candidate.low_cost_terminal_net_return > (
        candidate.primary_terminal_net_return
    )
    assert candidate.primary_terminal_net_return > (
        candidate.high_cost_terminal_net_return
    )
    selection = select_massive_adaptive_profit_checkpoint_v2((candidate,))
    assert selection.selected_epoch_index == 3
    assert not selection.development_checkpoint_selection_authorized
    assert not selection.outer_evaluation_authorized
    authority = materialize_massive_adaptive_profit_checkpoint_selection_authority_v2(
        root=tmp_path,
        artifact_id="synthetic-selection",
        candidates=(candidate,),
        committed_at_ms=1,
    )
    assert authority.runtime_selection_replayed
    assert not authority.development_checkpoint_selection_authorized
    generic = parse_massive_adaptive_profit_checkpoint_selection_authority_v2(
        root=tmp_path,
        loaded_source=authority.loaded_source,
    )
    assert not generic.runtime_selection_replayed
    assert generic.runtime_candidates is None
    replayed = authorize_massive_adaptive_profit_checkpoint_selection_authority_v2(
        root=tmp_path,
        authority=generic,
        candidates=(candidate,),
    )
    assert replayed.semantic_receipt_sha256 == authority.semantic_receipt_sha256
    changed = replace(candidate, primary_dollar_net_profit=123.0)
    changed = replace(
        changed,
        semantic_receipt_sha256=semantic_sha256(changed.semantic_unsigned()),
    )
    with pytest.raises(
        MassiveAdaptiveProfitCheckpointSelectionAuthorityV2Error,
        match="does not replay",
    ):
        authorize_massive_adaptive_profit_checkpoint_selection_authority_v2(
            root=tmp_path,
            authority=generic,
            candidates=(changed,),
        )
    with pytest.raises(
        MassiveAdaptiveOuterInferencePlanV1Error,
        match="source-qualified checkpoint selection",
    ):
        build_massive_adaptive_outer_inference_plan_v1(
            checkpoint_selection=authority,
            selected_checkpoint=checkpoint,
            decision_tensor=SimpleNamespace(validate=lambda: None),
            decision_roots=(),
            split_plan=SimpleNamespace(validate=lambda: None),
            fold_index=0,
            model_spec=SimpleNamespace(validate=lambda: None),
        )
    with pytest.raises(
        MassiveAdaptiveOuterForecastArchiveV1Error,
        match="frozen qualified selection",
    ):
        materialize_massive_adaptive_outer_forecast_archive_v1(
            root=tmp_path,
            artifact_id="forbidden-outer",
            checkpoint_selection=authority,
            selected_checkpoint=checkpoint,
            training_window_plan=SimpleNamespace(validate=lambda: None),
            outer_tensor=SimpleNamespace(validate=lambda: None),
            outer_decision_roots=(),
            outer_plan=SimpleNamespace(validate=lambda: None),
            model_spec=SimpleNamespace(validate=lambda: None),
            committed_at_ms=2,
        )


def test_compiler_authority_has_no_future_fill_input() -> None:
    import inspect

    from rl_quant.evaluation.massive_adaptive_compiler_input_authority_v1 import (
        build_massive_adaptive_compiler_input_authority_v1,
    )

    assert (
        "fill_source"
        not in inspect.signature(
            build_massive_adaptive_compiler_input_authority_v1
        ).parameters
    )


def test_training_calibration_generic_reload_is_nonauthorizing(tmp_path) -> None:
    value = _calibration(
        tuple(f"SEC-{index:03d}" for index in range(100)), root=tmp_path
    )
    assert value.runtime_calibration_replayed
    assert value.loaded_source is not None
    generic = parse_massive_adaptive_forecast_calibration_v1(
        root=tmp_path, loaded_source=value.loaded_source
    )
    assert not generic.runtime_calibration_replayed
    assert not generic.development_calibration_authorized


def test_calibration_v2_is_checkpoint_and_fold_bound() -> None:
    security_ids = tuple(f"SEC-{index:03d}" for index in range(100))
    values = _calibration_v2(security_ids)
    calibration = values.calibration

    assert calibration.fold_index == 0
    assert calibration.checkpoint_receipt_sha256 == (
        values.checkpoint.semantic_receipt_sha256
    )
    assert calibration.model_state_receipt_sha256 == (
        values.checkpoint.model_state_receipt_sha256
    )
    assert calibration.calibration_fit_stop_session_date == "2023-12-29"
    assert not calibration.development_calibration_authorized

    wrong_fold = SimpleNamespace(**values.training_window.__dict__)
    wrong_fold.fold_index = 1
    wrong_fold.semantic_receipt_sha256 = _digest("wrong-fold-window")
    with pytest.raises(
        MassiveAdaptiveForecastCalibrationV2Error,
        match="checkpoint, forecast, or training fold|identity",
    ):
        build_massive_adaptive_forecast_calibration_v2(
            checkpoint=values.checkpoint,
            training_forecasts=values.training_forecasts,
            training_targets=values.training_targets,
            training_window_plan=wrong_fold,
        )


def test_profit_trace_v2_starts_from_cash_and_uses_one_benchmark() -> None:
    fixture = _fixture()
    calibration_values = _calibration_v2(fixture.forecast_archive.runtime_rows[0].security_ids)
    fixture.forecast_archive.fold_index = 0
    fixture.forecast_archive.checkpoint_receipt_sha256 = (
        calibration_values.checkpoint.semantic_receipt_sha256
    )
    fixture.forecast_archive.model_state_receipt_sha256 = (
        calibration_values.checkpoint.model_state_receipt_sha256
    )
    fixture.forecast_archive.training_window_plan_receipt_sha256 = (
        calibration_values.training_window.semantic_receipt_sha256
    )
    fixture.inference_plan.fold_index = 0

    trace = build_massive_adaptive_profit_trace_v2(
        forecast_archive=fixture.forecast_archive,
        calibration=calibration_values.calibration,
        inference_plan=fixture.inference_plan,
        decision_roots=fixture.roots,
        context_origins=fixture.contexts,
        fill_source=fixture.fill_source,
        daily_input_authority=fixture.daily,
        identity_authority=fixture.identity,
        initial_capital=10_000_000.0,
    )

    assert trace.rows[0].pretrade_equity == 10_000_000.0
    assert trace.rows[0].benchmark_pretrade_equity == 10_000_000.0
    assert trace.rows[0].transaction_cost > 0.0
    assert trace.rows[0].turnover > 0.0
    assert len(trace.benchmark_authority_receipts) == len(trace.rows)
    assert trace.rows[1].pretrade_book_receipt_sha256 == (
        trace.rows[0].posttrade_book_receipt_sha256
    )
    assert trace.rows[1].benchmark_pretrade_book_receipt_sha256 == (
        trace.rows[0].benchmark_posttrade_book_receipt_sha256
    )
    assert not trace.source_data_qualified
    assert not trace.reinforcement_learning_authorized


def test_zero_rl_action_matches_deterministic_execution_and_reward() -> None:
    fixture = _fixture()
    calibration_values = _calibration_v2(
        fixture.forecast_archive.runtime_rows[0].security_ids
    )
    fixture.forecast_archive.fold_index = 0
    fixture.forecast_archive.checkpoint_receipt_sha256 = (
        calibration_values.checkpoint.semantic_receipt_sha256
    )
    fixture.forecast_archive.model_state_receipt_sha256 = (
        calibration_values.checkpoint.model_state_receipt_sha256
    )
    fixture.forecast_archive.training_window_plan_receipt_sha256 = (
        calibration_values.training_window.semantic_receipt_sha256
    )
    fixture.inference_plan.fold_index = 0
    plan_row = fixture.inference_plan.rows[0]
    forecast_row = fixture.forecast_archive.runtime_rows[0]
    initial = build_massive_adaptive_initial_book_authority_v1(
        decision_session_date=plan_row.decision_session_date,
        initial_capital=10_000_000.0,
        forecast_archive_receipt_sha256=fixture.forecast_archive.semantic_receipt_sha256,
        inference_plan_receipt_sha256=fixture.inference_plan.semantic_receipt_sha256,
        source_data_qualified=False,
    )
    prepared = prepare_massive_adaptive_economic_step_v1(
        forecast_archive=fixture.forecast_archive,
        forecast_row=forecast_row,
        calibration=calibration_values.calibration,
        inference_row=plan_row,
        decision_root=fixture.roots[0],
        context_origin=fixture.contexts[0],
        strategy_book=initial.strategy_book,
        neutral_book=initial.neutral_book,
        benchmark_book=initial.benchmark_book,
        daily_input_authority=fixture.daily,
        identity_authority=fixture.identity,
    )
    config = MassiveAdaptivePortfolioCompilerConfigV1()
    neutral_decision = compile_massive_adaptive_portfolio_v1(
        prepared.neutral_compiler_inputs,
        config=config,
    )
    action = neutral_massive_adaptive_rl_action_v1()
    control, policy_decision = compile_massive_adaptive_rl_control_v1(
        inputs=prepared.strategy_compiler_inputs,
        config=config,
        action=action,
    )
    step = settle_massive_adaptive_economic_step_v1(
        prepared=prepared,
        policy_decision=policy_decision,
        neutral_decision=neutral_decision,
        fill_source=fixture.fill_source,
        economic_event_archive=None,
        daily_input_authority=fixture.daily,
        identity_authority=fixture.identity,
        transaction_cost_basis_points=20.0,
        maximum_fill_participation=0.02,
        policy_action_receipt_sha256=action.semantic_receipt_sha256,
        policy_control_receipt_sha256=control.semantic_receipt_sha256,
    )

    assert step.neutral_equivalence
    assert policy_decision == neutral_decision
    assert step.strategy_execution == step.neutral_execution
    assert step.strategy_posttrade_book == step.neutral_posttrade_book
    assert step.incremental_rl_log_return == 0.0
    assert step.optimization_reward_basis_points == 0.0


def test_no_position_age_in_baseline_observation() -> None:
    fixture = _fixture()
    calibration_values = _calibration_v2(
        fixture.forecast_archive.runtime_rows[0].security_ids
    )
    fixture.forecast_archive.fold_index = 0
    fixture.forecast_archive.checkpoint_receipt_sha256 = (
        calibration_values.checkpoint.semantic_receipt_sha256
    )
    fixture.forecast_archive.model_state_receipt_sha256 = (
        calibration_values.checkpoint.model_state_receipt_sha256
    )
    fixture.forecast_archive.training_window_plan_receipt_sha256 = (
        calibration_values.training_window.semantic_receipt_sha256
    )
    fixture.inference_plan.fold_index = 0
    plan_row = fixture.inference_plan.rows[0]
    initial = build_massive_adaptive_initial_book_authority_v1(
        decision_session_date=plan_row.decision_session_date,
        initial_capital=10_000_000.0,
        forecast_archive_receipt_sha256=fixture.forecast_archive.semantic_receipt_sha256,
        inference_plan_receipt_sha256=fixture.inference_plan.semantic_receipt_sha256,
        source_data_qualified=False,
    )
    prepared = prepare_massive_adaptive_economic_step_v1(
        forecast_archive=fixture.forecast_archive,
        forecast_row=fixture.forecast_archive.runtime_rows[0],
        calibration=calibration_values.calibration,
        inference_row=plan_row,
        decision_root=fixture.roots[0],
        context_origin=fixture.contexts[0],
        strategy_book=initial.strategy_book,
        neutral_book=initial.neutral_book,
        benchmark_book=initial.benchmark_book,
        daily_input_authority=fixture.daily,
        identity_authority=fixture.identity,
    )
    observation = build_massive_adaptive_rl_observation_v1(
        prepared=prepared,
        previous_action=neutral_massive_adaptive_rl_action_v1(),
        trailing_state=MassiveAdaptiveRLTrailingStateV1(),
    )

    forbidden = ("age", "duration", "persistence", "hazard", "scheduled_exit")
    assert len(observation.values) <= 128
    assert all(
        not any(fragment in name for fragment in forbidden)
        for name in observation.feature_names
    )
    assert all(
        not any(fragment in field.name for fragment in forbidden)
        for field in fields(MassiveAdaptiveRLObservationV1)
    )


def _adaptive_env_fixture():
    fixture = _fixture()
    calibration_values = _calibration_v2(
        fixture.forecast_archive.runtime_rows[0].security_ids
    )
    fixture.forecast_archive.fold_index = 0
    fixture.forecast_archive.checkpoint_receipt_sha256 = (
        calibration_values.checkpoint.semantic_receipt_sha256
    )
    fixture.forecast_archive.model_state_receipt_sha256 = (
        calibration_values.checkpoint.model_state_receipt_sha256
    )
    fixture.forecast_archive.training_window_plan_receipt_sha256 = (
        calibration_values.training_window.semantic_receipt_sha256
    )
    fixture.inference_plan.fold_index = 0
    environment = MassiveAdaptiveProfitabilityEnvV1(
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
    )
    return fixture, calibration_values, environment


def test_rl_rollout_truncation_preserves_all_books_and_resume() -> None:
    fixture, calibration_values, environment = _adaptive_env_fixture()
    observation, _ = environment.reset()
    action = neutral_massive_adaptive_rl_action_v1()
    next_observation, reward, terminated, truncated, info = environment.step(action)
    assert observation.values
    assert next_observation is not None
    assert reward == 0.0
    assert not terminated
    assert not truncated
    transition = info["transition"]
    assert isinstance(transition, MassiveAdaptiveRLTransitionV1)
    continuation = environment.rollout_boundary_state()
    assert continuation == environment.state

    resumed = MassiveAdaptiveProfitabilityEnvV1(
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
    )
    resumed.restore(continuation)
    uninterrupted_result = environment.step(action)
    resumed_result = resumed.step(action)
    assert uninterrupted_result == resumed_result
    assert uninterrupted_result[2]
    terminal_transition = uninterrupted_result[4]["transition"]
    assert terminal_transition.strategy_terminal_liquidation_cost > 0.0
    assert terminal_transition.incremental_rl_log_return == 0.0


def test_no_scheduled_exit_in_environment() -> None:
    forbidden = (
        "position_age",
        "holding_period",
        "duration_reward",
        "persistence_bonus",
        "scheduled_exit",
        "hazard",
        "cohort",
    )
    for record_type in (
        MassiveAdaptiveProfitabilityEnvStateV1,
        MassiveAdaptiveRLTransitionV1,
    ):
        assert all(
            not any(fragment in field.name for fragment in forbidden)
            for field in fields(record_type)
        )
    source = adaptive_env_module.__file__
    assert source is not None
    source_text = Path(source).read_text(encoding="utf-8")
    assert not any(fragment in source_text for fragment in forbidden)


def test_no_duration_term_in_reward() -> None:
    assert (
        MASSIVE_ADAPTIVE_PROFITABILITY_ENV_V1_SPEC_SHA256
        == adaptive_env_module.MASSIVE_ADAPTIVE_PROFITABILITY_ENV_V1_SPEC_SHA256
    )
    source = adaptive_env_module.__file__
    assert source is not None
    source_text = Path(source).read_text(encoding="utf-8")
    forbidden = (
        "holding_reward",
        "persistence_bonus",
        "early_sale_penalty",
        "completion_bonus",
        "duration_regularization",
    )
    assert not any(fragment in source_text for fragment in forbidden)


def test_adaptive_ppo_checkpoint_resume_is_exact() -> None:
    _, _, environment = _adaptive_env_fixture()
    config = MassiveAdaptivePPOConfigV1(
        rollout_length=2,
        minibatch_size=2,
        epochs_per_rollout=2,
        seed=71,
    )
    trainer = MassiveAdaptivePPOTrainerV1(
        environment=environment,
        model=MassiveAdaptivePPOActorCriticV1(observation_dim=90),
        config=config,
    )
    first_rollout = trainer.collect_rollout()
    trainer.update(first_rollout)
    checkpoint = trainer.checkpoint()
    uninterrupted_rollout = trainer.collect_rollout()
    uninterrupted_metrics = trainer.update(uninterrupted_rollout)
    uninterrupted_checkpoint = trainer.checkpoint()

    _, _, resumed_environment = _adaptive_env_fixture()
    resumed = MassiveAdaptivePPOTrainerV1(
        environment=resumed_environment,
        model=MassiveAdaptivePPOActorCriticV1(observation_dim=90),
        config=config,
    )
    resumed.restore(checkpoint)
    resumed_rollout = resumed.collect_rollout()
    resumed_metrics = resumed.update(resumed_rollout)
    resumed_checkpoint = resumed.checkpoint()

    assert torch.equal(uninterrupted_rollout.actions, resumed_rollout.actions)
    assert uninterrupted_rollout.transition_receipts == (
        resumed_rollout.transition_receipts
    )
    assert uninterrupted_metrics == resumed_metrics
    assert uninterrupted_checkpoint.semantic_receipt_sha256 == (
        resumed_checkpoint.semantic_receipt_sha256
    )
    assert all(
        torch.equal(value, resumed_checkpoint.model_state[name])
        for name, value in uninterrupted_checkpoint.model_state.items()
    )

    falsely_qualified_duplicate_dates = replace(
        uninterrupted_checkpoint,
        training_forecast_authority_receipt_sha256=_digest(
            "qualified-training-forecast"
        ),
        fit_environment_authority_receipts=(_digest("qualified-fit-environment"),),
        transition_source_data_qualified=tuple(
            True for _ in uninterrupted_checkpoint.transition_receipts
        ),
        source_data_qualified=True,
        development_rl_training_authorized=True,
        semantic_receipt_sha256="0" * 64,
    )
    falsely_qualified_duplicate_dates = replace(
        falsely_qualified_duplicate_dates,
        semantic_receipt_sha256=semantic_sha256(
            falsely_qualified_duplicate_dates.semantic_unsigned()
        ),
    )
    with pytest.raises(MassiveAdaptivePPOV1Error, match="checkpoint differs"):
        falsely_qualified_duplicate_dates.validate()


def test_adaptive_ppo_checkpoint_is_durable_and_runtime_stripped(tmp_path) -> None:
    _, _, environment = _adaptive_env_fixture()
    config = MassiveAdaptivePPOConfigV1(
        rollout_length=2,
        minibatch_size=2,
        epochs_per_rollout=1,
        seed=73,
    )
    trainer = MassiveAdaptivePPOTrainerV1(
        environment=environment,
        model=MassiveAdaptivePPOActorCriticV1(observation_dim=90),
        config=config,
    )
    trainer.update(trainer.collect_rollout())
    checkpoint = trainer.checkpoint()
    training_receipt = _digest("durable-rl-training-forecast-authority")
    checkpoint = replace(
        checkpoint,
        training_forecast_authority_receipt_sha256=training_receipt,
        semantic_receipt_sha256="0" * 64,
    )
    checkpoint = replace(
        checkpoint,
        semantic_receipt_sha256=semantic_sha256(checkpoint.semantic_unsigned()),
    )
    checkpoint.validate()
    training_authority = SimpleNamespace(
        validate=lambda: None,
        semantic_receipt_sha256=training_receipt,
        source_data_qualified=False,
        reinforcement_learning_authorized=False,
    )

    authority = materialize_massive_adaptive_rl_checkpoint_authority_v1(
        root=tmp_path,
        artifact_id="durable-rl-checkpoint",
        checkpoint=checkpoint,
        training_forecast_authority=training_authority,
        committed_at_ms=90_000,
    )
    assert authority.runtime_checkpoint_replayed
    assert authority.runtime_checkpoint is not None
    assert authority.runtime_checkpoint.semantic_receipt_sha256 == (
        checkpoint.semantic_receipt_sha256
    )
    assert not authority.exact_resume_authorized

    generic = parse_massive_adaptive_rl_checkpoint_authority_v1(
        root=tmp_path,
        loaded_source=authority.loaded_source,
    )
    assert generic.runtime_checkpoint is None
    assert not generic.runtime_checkpoint_replayed
    replayed = authorize_massive_adaptive_rl_checkpoint_authority_v1(
        root=tmp_path,
        authority=generic,
        training_forecast_authority=training_authority,
    )
    assert replayed.semantic_receipt_sha256 == authority.semantic_receipt_sha256
    assert replayed.runtime_checkpoint is not None
    assert replayed.runtime_checkpoint.semantic_receipt_sha256 == (
        checkpoint.semantic_receipt_sha256
    )


def test_policy_trace_is_derived_from_complete_economic_transitions() -> None:
    fixture, calibration_values, environment = _adaptive_env_fixture()
    environment.reset()
    transitions = []
    terminated = False
    while not terminated:
        _, _, terminated, _, info = environment.step(
            neutral_massive_adaptive_rl_action_v1()
        )
        transitions.append(info["transition"])

    trainer_environment = _adaptive_env_fixture()[2]
    trainer_environment.reset()
    trainer = MassiveAdaptivePPOTrainerV1(
        environment=trainer_environment,
        model=MassiveAdaptivePPOActorCriticV1(observation_dim=90),
    )
    checkpoint = trainer.checkpoint()
    training_receipt = _digest("trace-rl-training-forecast-authority")
    checkpoint = replace(
        checkpoint,
        training_forecast_authority_receipt_sha256=training_receipt,
        semantic_receipt_sha256="0" * 64,
    )
    checkpoint = replace(
        checkpoint,
        semantic_receipt_sha256=semantic_sha256(checkpoint.semantic_unsigned()),
    )
    trace = build_massive_adaptive_rl_policy_trace_v1(
        fold_index=0,
        checkpoint=checkpoint,
        forecast_archive_receipt_sha256=(
            fixture.forecast_archive.semantic_receipt_sha256
        ),
        inference_plan_receipt_sha256=fixture.inference_plan.semantic_receipt_sha256,
        calibration_receipt_sha256=(
            calibration_values.calibration.semantic_receipt_sha256
        ),
        transaction_cost_basis_points=20.0,
        initial_capital=10_000_000.0,
        transitions=transitions,
        frozen_targets_replayed=False,
    )

    assert trace.transition_receipts == tuple(
        transition.semantic_receipt_sha256 for transition in transitions
    )
    assert trace.cumulative_incremental_rl_log_return == 0.0
    assert trace.terminal_liquidation_adjusted_return < 0.0
    assert not trace.source_data_qualified
    assert not trace.profitability_reporting_authorized


def test_full_portfolio_turnover_includes_cash_leg() -> None:
    execution = SimpleNamespace(
        rows=(
            SimpleNamespace(filled_shares=1.0, executed_notional=80.0),
            SimpleNamespace(filled_shares=-1.0, executed_notional=20.0),
        )
    )
    assert full_portfolio_one_way_turnover_v2(execution, 1_000.0) == 0.08


def test_outer_forecast_rejects_checkpoint_from_another_fold() -> None:
    checkpoint = SimpleNamespace(
        validate=lambda: None,
        window_plan_receipt_sha256=_digest("fold-zero-training-window"),
    )
    training_window = SimpleNamespace(
        validate=lambda: None,
        split_role="training",
        fold_index=0,
        semantic_receipt_sha256=_digest("fold-zero-training-window"),
    )
    outer_plan = SimpleNamespace(validate=lambda: None, fold_index=1)

    with pytest.raises(
        MassiveAdaptiveOuterForecastArchiveV1Error,
        match="outer fold training window differ",
    ):
        validate_massive_adaptive_outer_fold_binding_v1(
            selected_checkpoint=checkpoint,
            training_window_plan=training_window,
            outer_plan=outer_plan,
        )
